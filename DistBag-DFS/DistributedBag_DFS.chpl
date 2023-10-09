/*
  A highly parallel segmented multi-pool. Each node gets its own bag, and each
  bag is segmented into 'here.maxTaskPar' segments. Segments allow for actual
  parallelism while operating in that it enables us to manage 'best-case',
  'average-case', and 'worst-case' scenarios by making multiple passes over each
  segment. In the case where there is no oversubscription, the best-case will
  always be achieved (considering any other conditions are also met), while in
  the case of oversubscription or, for example a near empty bag, we fall into
  the 'average-case', etc. Examples of 'best-case' scenarios for a removal would
  be when a segment contains some elements we can drain, and the 'average-case'
  would be to find any segment that contains elements we can drain (local work
  stealing), and so on.
  This data structure also employs a bi-level work stealing algorithm (WS) that
  first tries to steal a ratio of elements across local segments, and if it fails,
  it tries a global steal across segments of another bag instance. We steal a ratio
  of elements, say 25%, because it leaves all victim segments with 75% of their
  work; the more elements they have, the more we take, but the less they have,
  the less we steal; this also has the added benefit of reducing unnecessary WS
  between segments when the bag is nearly emptied. Stealing ensures that all segments
  remain filled and that we still achieve parallelism across segments for removal
  operations. Lastly, we attempt to steal a maximum of `N / sizeof(eltType)`, where
  N is some size in megabytes (representing how much data can be sent in one network
  request), which keeps down excessive communication.
  This data structure does not come without flaws; as WS is dynamic and triggered
  on demand, WS can still be performed in excess, which dramatically causes a
  performance drop. Furthermore, a severe imbalance across nodes, such as an unfair
  distribution of work to a single or small subset of nodes, may also causes an
  equally severe performance degradation. This data structure scales in terms of
  nodes, processors per node, and even work load. The larger the work load, the
  more data that gets stolen when WS, and better locality of elements  among segments.
  As well, to achieve true parallelism, usage of a privatized instance is a requirement,
  as it avoids the overhead of remotely accessing class fields, bounding performance
  on communication.
*/

/* Implements a highly parallel segmented multi-pool.
  Summary
  _______
  A parallel-safe distributed multi-pool implementation that scales in terms of
  nodes, processors per node (PPN), and workload; The more PPN, the more segments
  we allocate to increase raw parallelism, and the larger the workload the better
  locality (see: const:`distributedBagInitialBlockCap`). This data structure is
  unordered and employs its own work stealing algorithm to balance work across nodes.
  .. note::
    This module is a work in progress and may change in future releases.
  Usage
  _____
  To use: record:`DistBag_DFS`, the initializer must be invoked explicitly to
  properly initialize the structure. Using the default state without initializing
  will result in a halt.
  .. code-block:: chapel
    var bag = new DistBag_DFS(int, targetLocales=ourTargetLocales);
  While the bag is safe to use in a distributed manner, each node always operates
  on its privatized instance. This means that it is easy to add data in bulk, expecting
  it to be distributed, when in reality it is not; if another node needs data, it
  will steal work on-demand.
  .. code-block:: chapel
    bag.addBulk(1..N);
    bag.balance();
  Methods
  _______
*/

module DistributedBag_DFS
{
  public use Collection;
  private use IO;

  use Random;
  use List;
  use Math;

  /*
    The phases for operations. An operation is composed of multiple phases,
    where they make a full pass searching for ideal conditions, then less-than-ideal
    conditions; this is required to ensure maximized parallelism at all times, and
    critical to good performance, especially when a node is oversubscribed.
  */
  private param REMOVE_SIMPLE       = 1;
  private param REMOVE_LOCAL_STEAL  = 2;
  private param REMOVE_GLOBAL_STEAL = 3;
  private param PERFORMANCE_PATCH   = 4;

  private param REMOVE_SUCCESS   =  1;
  private param REMOVE_FAST_EXIT =  0;
  private param REMOVE_FAIL      = -1;

  /*
    The initial amount of elements in an unroll block. Each successive unroll block
    is double the size of it's predecessor, allowing for better locality for when
    there are larger numbers of elements. The better the locality, the better raw
    performance and easier it is to redistribute work.
  */
  config const distributedBagInitialBlockCap: int = 1024;
  /*
    To prevent stealing too many elements (horizontally) from another node's segment
    (hence creating an artificial load imbalance), if the other node's segment has
    less than a certain threshold (see :const:`distributedBagWorkStealingMemCap`) but above
    another threshold (see :const:`distributedBagWorkStealingMinElems`), we steal a percentage of their
    elements, leaving them with majority of their elements. This way, the amount the
    other segment loses is proportional to how much it owns, ensuring a balance.
  */
  config const distributedBagWorkStealingRatio: real = 0.25;
  /*
    The maximum amount of work to steal from a horizontal node's segment. This
    should be set to a value, in megabytes, that determines the maximum amount of
    data that should be sent in bulk at once. The maximum number of elements is
    determined by: (:const:`distributedBagWorkStealingMemCap` * 1024 * 1024) / sizeof(eltType).
    For example, if we are storing 8-byte integers and have a 1MB limit, we would
    have a maximum of 125,000 elements stolen at once.
  */
  config const distributedBagWorkStealingMemCap: real = 1.0;
  /*
    The minimum number of elements a horizontal segment must have to become eligible
    to be stolen from. This may be useful if some segments produce less elements than
    others and should not be stolen from.
  */
  config const distributedBagWorkStealingMinElts: int = 1;
  /*
    The maximum amount of elements in an unroll block. This is crucial to ensure memory
    usage does not rapidly grow out of control.
  */
  config const distributedBagMaxBlockCap: int = 1024 * 1024;

  /*
    Reference counter for DistributedBag_DFS
  */
  @chpldoc.nodoc
  class DistributedBagRC
  {
    type eltType;
    var _pid: int;

    proc deinit()
    {
      coforall loc in Locales do on loc {
        delete chpl_getPrivatizedCopy(unmanaged DistributedBagImpl(eltType), _pid);
      }
    }
  }

  /*
    A parallel-safe distributed multiset implementation that scales in terms of
    nodes, processors per node (PPN), and workload; The more PPN, the more segments
    we allocate to increase raw parallelism, and the larger the workload the better
    locality (see :const:`distributedBagInitialBlockCap`). This data structure is unordered and employs
    its own work-stealing algorithm, and provides a means to obtain a privatized instance of
    the data structure for maximized performance.
  */
  pragma "always RVF"
  record DistBag_DFS : serializable
  {
    type eltType;

    // This is unused, and merely for documentation purposes. See '_value'.
    /*
      The implementation of the Bag is forwarded. See :class:`DistributedBagImpl` for
      documentation.
    */
    var _impl: unmanaged DistributedBagImpl(eltType)?;

    // Privatized id
    @chpldoc.nodoc
    var _pid: int = -1;

    // Reference Counting
    @chpldoc.nodoc
    var _rc: shared DistributedBagRC(eltType);

    @chpldoc.nodoc
    proc init(type eltType, targetLocales = Locales)
    {
      this.eltType = eltType;
      this._pid = (new unmanaged DistributedBagImpl(eltType, targetLocales = targetLocales)).pid;
      this._rc = new shared DistributedBagRC(eltType, _pid = _pid);
    }

    @chpldoc.nodoc
    inline proc _value
    {
      if (_pid == -1) then halt("DistBag_DFS is uninitialized.");
      return chpl_getPrivatizedCopy(unmanaged DistributedBagImpl(eltType), _pid);
    }

    @chpldoc.nodoc
    proc readThis(f) throws {
      compilerError("Reading a DistBag_DFS is not supported");
    }

    @chpldoc.nodoc
    proc deserialize(reader, ref deserializer) throws {
      compilerError("Reading a DistBag is not supported");
    }

    @chpldoc.nodoc
    proc init(type eltType, reader: fileReader(?), ref deserializer) {
      this.init(eltType);
      compilerError("Deserializing a DistBag is not yet supported");
    }

    // Write the contents of DistBag_DFS to a channel.
    @chpldoc.nodoc
    proc writeThis(ch) throws {
      ch.write("[");
      var size = this.getSize();
      for (i, iteration) in zip(this, 0..<size) {
        ch.write(i);
        if (iteration < size-1) then ch.write(", ");
      }
      ch.write("]");
    }

    @chpldoc.nodoc
    proc serialize(writer, ref serializer) throws {
      writeThis(writer);
    }

    forwarding _value;
  } // end 'DistBag_DFS' record

  class DistributedBagImpl : CollectionImpl(?)
  {
    @chpldoc.nodoc
    var targetLocDom: domain(1);

    /*
      The locales to allocate bags for and load balance across.
    */
    var targetLocales: [targetLocDom] locale;

    @chpldoc.nodoc
    var pid: int = -1;

    // Node-local fields below. These fields are specific to the privatized instance.
    // To access them from another node, make sure you use 'getPrivatizedThis'
    @chpldoc.nodoc
    var bag: unmanaged Bag(eltType)?;

    proc init(type eltType, targetLocales: [?targetLocDom] locale = Locales)
    {
      super.init(eltType);

      this.targetLocDom  = targetLocDom;
      this.targetLocales = targetLocales;

      init this;

      this.pid = _newPrivatizedClass(this);
      this.bag = new unmanaged Bag(eltType, this);
    }

    @chpldoc.nodoc
    proc init(other, pid, type eltType = other.eltType)
    {
      super.init(eltType);

      this.targetLocDom  = other.targetLocDom;
      this.targetLocales = other.targetLocales;
      this.pid           = pid;

      init this;

      this.bag = new unmanaged Bag(eltType, this);
    }

    @chpldoc.nodoc
    proc deinit()
    {
      delete bag;
    }

    @chpldoc.nodoc
    proc dsiPrivatize(pid)
    {
      return new unmanaged DistributedBagImpl(this, pid);
    }

    @chpldoc.nodoc
    proc dsiGetPrivatizeData()
    {
      return pid;
    }

    @chpldoc.nodoc
    inline proc getPrivatizedThis
    {
      return chpl_getPrivatizedCopy(this.type, pid);
    }

    @chpldoc.nodoc
    iter targetLocalesNotHere()
    {
      foreach loc in targetLocales {
        if (loc != here) then yield loc;
      }
    }

    /*
      Insert an element in the calling task's segment of this node's bag.
    */
    proc add(elt: eltType, taskId: int): bool
    {
      return bag!.add(elt, taskId);
    }

    /*
      Insert elements in bulk in the calling thread's segment of this node's bag.
      If the node's bag rejects an element, we cease to offer more. We return the
      number of elements successfully added to this data structure.
    */
    proc addBulk(elts, taskId: int): int
    {
      return bag!.addBulk(elts, taskId);
    }

    /*
      Remove an element from the calling thread's segment of this node's bag.
      If the thread's segment is empty, it will attempt to steal an element from
      the segment of another thread or node.
    */
    proc remove(taskId: int): (int, eltType)
    {
      return bag!.remove(taskId);
    }

    proc removeBulk_(n: int, taskId: int): (bool, [0..#n] eltType)
    {
      return bag!.removeBulk_(n, taskId);
    }

    /*
      Obtain the number of elements held in all bags across all nodes. This method
      is best-effort and can be non-deterministic for concurrent updates across nodes,
      and may miss elements or even count duplicates resulting from any concurrent
      insertion or removal operations.
    */
    override proc getSize(): int
    {
      var size: atomic int;
      coforall loc in targetLocales do on loc {
        var instance = getPrivatizedThis;
        forall taskId in 0..#here.maxTaskPar do
          size.add(instance.bag!.segments[taskId].nElts);
      }

      return size.read();
    }

    /*
      Performs a lookup to determine if the requested element exists in this bag.
      This method is best-effort and can be non-deterministic for concurrent
      updates across nodes, and may miss elements resulting from any concurrent
      insertion or removal operations.
    */
    override proc contains(elt: eltType): bool
    {
      var foundElt: atomic bool;
      forall elem in getPrivatizedThis {
        if (elem == elt) then foundElt.write(true);
      }

      return foundElt.read();
    }

    /*
      Clear all bags across all nodes in a best-effort approach. Elements added or
      moved around from concurrent additions or removals may be missed while clearing.
    */
    override proc clear(): void
    {
      coforall loc in targetLocales do on loc {
        var instance = getPrivatizedThis;
        forall taskId in 0..#here.maxTaskPar {
          ref segment = instance.bag!.segments[taskId];

          segment.lock_block.readFE();

          delete segment.block;
          segment.block = new unmanaged Block(eltType, distributedBagInitialBlockCap);
          segment.nElts_shared.write(0);
          segment.head.write(0);
          segment.split.write(0);
          segment.globalSteal.write(false);
          segment.split_request.write(false);
          segment.lock.writeXF(true);
          segment.lock_n.writeXF(true);
          segment.tail = 0;
          segment.o_split = 0;

          segment.lock_block.writeEF(true);
        }
        instance.bag!.globalStealInProgress.write(false);
      }
    }

    /*
      Triggers a more static approach to load balancing, fairly redistributing all
      elements fairly for bags across nodes. The result will result in all segments
      having roughly the same amount of elements.
    */
    // TODO: is 'balance' needed?

    /*
      Iterate over each bag in each node. To avoid holding onto locks, we take
      a snapshot approach, increasing memory consumption but also increasing parallelism.
      This allows other concurrent, even mutating, operations while iterating,
      but opens the possibility to iterating over duplicates or missing elements
      from concurrent operations.
      .. note::
      `zip` iteration is not yet supported with rectangular data structures.
      .. warning::
      Iteration takes a snapshot approach, and as such can easily result in a
      Out-Of-Memory issue. If the data structure is large, the user is doubly advised to use
      parallel iteration, for both performance and memory benefit.
    */
    override iter these(): eltType
    {
      for loc in targetLocales {
        for taskId in 0..#here.maxTaskPar {
          // The size of the snapshot is only known once we have the lock.
          var dom: domain(1) = {0..-1};
          var buffer: [dom] eltType;
          on loc {
            ref segment = getPrivatizedThis.bag!.segments[taskId];

              dom = {0..#segment.nElts};
              for i in dom {
                buffer[i] = segment.block.elts[segment.block.headId + i];
              }
          }
          // Process this chunk if we have one...
          foreach elt in buffer {
            yield elt;
          }
        }
      }
    }

    // UNUSED (these)
    /* iter these(param tag : iterKind) where tag == iterKind.leader
    {
      coforall loc in targetLocales do on loc {
        var instance = getPrivatizedThis;
        coforall segmentIdx in 0..#here.maxTaskPar {
          ref segment = instance.bag!.segments[segmentIdx];
          if segment.acquireIfNonEmpty(STATUS_LOOKUP) {
            // Create a snapshot...
            var block = segment.headBlock;
            var bufferSz = segment.nElems.read():int;
            var buffer = c_malloc(eltType, bufferSz);
            var bufferOffset = 0;
            while (block != nil) {
              if (bufferOffset + block!.size > bufferSz) {
                halt("DistributedBag Internal Error: Snapshot attempt with bufferSz(", bufferSz, ") with offset bufferOffset(", bufferOffset + block!.size, ")");
              }
              __primitive("chpl_comm_array_put", block!.elems[0], here.id, buffer[bufferOffset], block!.size);
              bufferOffset += block!.size;
              block = block!.next;
            }
            // Yield this chunk to be process...
            segment.releaseStatus();
            yield (bufferSz, buffer);
            c_free(buffer);
          }
        }
      }
    } */

    // UNUSED (these)
    /* iter these(param tag : iterKind, followThis) where tag == iterKind.follower
    {
      var (bufferSz, buffer) = followThis;
      foreach i in 0..#bufferSz {
        yield buffer[i];
      }
    } */
  } // end 'DistributedBagImpl' class

  /*
    We maintain a multiset 'bag' per node. Each bag keeps a handle to it's parent,
    which is required for work stealing.
  */
  @chpldoc.nodoc
  class Bag
  {
    type eltType;

    // A handle to our parent 'distributed' bag, which is needed for work stealing.
    var parentHandle: borrowed DistributedBagImpl(eltType);

    var segments: [0..#here.maxTaskPar] Segment(eltType);

    var globalStealInProgress: atomic bool = false;

    proc init(type eltType, parentHandle)
    {
      this.eltType = eltType;
      this.parentHandle = parentHandle;
      // KNOWN ISSUE: 'this.complete' produces an error when 'eltType' is a Chapel
      // array (see Github issue #19859)
    }

    proc deinit()
    {
      forall segment in segments do
        delete segment.block;
    }

    /*
      Insertion operation.
    */
    proc add(elt: eltType, const taskId: int): bool
    {
      return segments[taskId].addElement(elt);
    }

    /*
      Insertion operation in bulk.
    */
    proc addBulk(elts, const taskId: int): int
    {
      return segments[taskId].addElements(elts);
    }

    /*
      This iterator is intented to select victim(s) in work-stealing strategies,
      according to the specified policy. By default, the 'rand' strategy is chosen and
      the calling thread/locale cannot be chosen. We can specify how many tries we want,
      by default, only 1 is performed.
    */
    iter victim(const N: int, const callerId: int, const policy: string = "rand", const tries: int = 1): int
    {
      var count: int;
      var limit: int = if (callerId == -1) then N else N-1;

      select policy {
        // In the 'ring' strategy, victims are selected in a round-robin fashion.
        when "ring" {
          var id = (callerId + 1) % N;

          while ((count < limit) && (count < tries)) {
            yield id;
            count += 1;
            id = (id + 1) % N;
          }
        }
        // In the 'rand' strategy, victims are randomly selected.
        when "rand" {
          var id: int;
          var victims: [0..#N] int = noinit;
          permutation(victims);

          while ((count < limit) && (count < tries)) {
            if (victims[id] != callerId) {
              yield victims[id];
              count += 1;
            }
            id += 1;
          }
        }
        otherwise halt("DistributedBag_DFS internal error: Wrong victim choice policy");
      }
    }

    /*
      Retrieval operation that succeeds when one of the three successives case
      succeeds. In BEST CASE, the caller try to remove an element from its segment.
      In AVERAGE CASE, the caller try to steal another segment of its bag instance.
      In WORST CASE, the caller try to steal another segment of another bag instance.
      The operation fails if all cases failed.
    */
    proc remove(const taskId: int): (int, eltType)
    {
      var phase = REMOVE_SIMPLE;
      if (numLocales > 1) then phase = PERFORMANCE_PATCH;

      ref segment = segments[taskId];
      var default: eltType;

      while true {
        select phase {
          /*
            Without this patch, the WS mechanism is not able to perform well on
            distributed settings. This could be explained by some bottlenecks, as
            well as mix-up between priority levels of local and global steals.

            TODO: investigate this in order to remove the patch.
          */
          when PERFORMANCE_PATCH {
            phase = REMOVE_SIMPLE;
          }

          /*
            SIMPLE:
            We try to retrieve an element in segment 'threadId'. Retrieval is done
            at the tail of the segment's block. This try fails if the private region
            is empty.
          */
          when REMOVE_SIMPLE {
            // if the private region contains at least one element to be removed...
            if (segment.nElts_private > 0) {
              // attempt to remove an element
              var (hasWork, elt): (bool, eltType) = segment.takeElement();

              if hasWork then return (REMOVE_SUCCESS, elt);
              /* if hasWork then return (1, elt);
              else return (-1, default); */
            }

            phase = REMOVE_LOCAL_STEAL;
          }

          /*
            LOCAL STEAL: intra-node work stealing
            It seems that segment 'threadId' is empty so we try to steal another one.
            The victim selection is set in the 'victim' iterator, and defaults to random.
            The work stealing fails when all segments don't satisfied the condition to be
            a victim, or when a shared region becomes empty due to a concurrent operation.
          */
          when REMOVE_LOCAL_STEAL {
            var splitreq: bool = false;

            // fast exit if: (1) a segment of our bag instance is doing a global steal
            // (2) a global steal is performed on our segment (priority is given).
            // if globalStealInProgress.read() then return (REMOVE_FAST_EXIT, default);

            if globalStealInProgress.read() {
              return (REMOVE_FAST_EXIT, default);
            }

            // selection of the victim segment
            for victimTaskId in victim(here.maxTaskPar, taskId, "rand", here.maxTaskPar) {
              ref targetSegment = segments[victimTaskId];

              if !targetSegment.globalSteal.read() {
                targetSegment.lock_block.readFE();
                // if the shared region contains enough elements to be stolen...
                if (distributedBagWorkStealingMinElts <= targetSegment.nElts_shared.read()) {
                  // attempt to steal an element
                  var (hasElt, elt): (bool, eltType) = targetSegment.stealElement();

                  // if the steal succeeds, we return, otherwise we continue
                  if hasElt {
                    targetSegment.lock_block.writeEF(true);
                    return (REMOVE_SUCCESS, elt);
                  }
                }
                // otherwise, if the private region has elements, we request for a split shifting
                else if (targetSegment.nElts_private > 1) {
                  splitreq = true;
                  targetSegment.split_request.write(true);
                }
                targetSegment.lock_block.writeEF(true);
              }
            }

            if splitreq then return (REMOVE_FAST_EXIT, default);

            phase = REMOVE_GLOBAL_STEAL;
          }

          /*
            GLOBAL STEAL: inter-node work stealing
            The caller fails to remove an element on its bag instance, so we try
            to steal another bag. The victim selection of the locale is set in the
            'victim' iterator, and the segments are then visited in a 'ring' fashion.
            The inter-node work stealing fails when all segments of all locales don't
            satisfied the condition to be a victim, or when a shared region becomes
            empty due to a concurrent operation.
          */
          when REMOVE_GLOBAL_STEAL {
            // fast exit for single-node execution
            if (numLocales == 1) then return (REMOVE_FAIL, default);

            // "Lock" the global steal operation
            if !globalStealInProgress.compareAndSwap(false, true) {
              return (REMOVE_FAST_EXIT, default);
            }

            const parentPid = parentHandle.pid;
            var stolenElts: list(eltType);

            // selection of the victim locale
            for victimLocaleId in victim(numLocales, here.id, "rand", 1) { //numLocales-1) {
              on Locales[victimLocaleId] {
                var targetBag = chpl_getPrivatizedCopy(parentHandle.type, parentPid).bag;
                // selection of the victim segment
                for victimTaskId in victim(here.maxTaskPar, -1, "rand", here.maxTaskPar) { //0..#here.maxTaskPar {
                  ref targetSegment = targetBag!.segments[victimTaskId];

                  targetSegment.globalSteal.write(true);

                  //var sharedElts: int = targetSegment.nElems_shared.read();
                  // if the shared region contains enough elements to be stolen...
                  targetSegment.lock_block.readFE();
                  if (1 < targetSegment.nElts_shared.read()) {
                    //for i in 0..#(targetSegment.nElems_shared.read()/2):int {
                      // attempt to steal an element
                      var (hasElt, elt): (bool, eltType) = targetSegment.stealElement();

                      // if the steal succeeds...
                      if hasElt {
                        stolenElts.insert(0, elt);
                      }
                  //  }
                  }
                  // otherwise, if the private region has elements, we request for a split shifting
                  else if (targetSegment.nElts_private > 1) {
                    targetSegment.split_request.write(true);
                  }

                  targetSegment.lock_block.writeEF(true);
                  targetSegment.globalSteal.write(false);
                }
              }
            }

            // if the global steal fails...
            if (stolenElts.size == 0) {
              // "Unlock" the global steal operation
              globalStealInProgress.write(false);
              return (REMOVE_FAIL, default);
            }
            else {
              segment.addElements(stolenElts);
              //segments[taskId].split.add((3*stolenElts.size/4):int);

              // "Unlock" the global steal operation
              globalStealInProgress.write(false);
              return (REMOVE_SUCCESS, segment.takeElement()[1]);
            }
          }

          otherwise do halt("DistributedBag_DFS Internal Error: Invalid phase #", phase);
        }
        currentTask.yieldExecution();
      }

      halt("DistributedBag_DFS Internal Error: DEADCODE.");
    }

    // This procedure is only callable by the owner thread
    proc removeBulk_(const n: int, const taskId: int)//: (bool, [] eltType)
    {
      ref segment = segments[taskId];
      var default: [0..-1] eltType;

      // if the private region contains enough elements to be removed...
      if (segment.tail - segment.o_split >= n) {
        var elts: [0..#n] eltType;
        for elt in elts do elt = segment.takeElement()[1];

        return (true, elts);
      }

      return (false, default);
    }
  } // end 'Bag' class

  /*
    A Segment is, in and of itself an unrolled linked list. We maintain one per core
    to ensure maximum parallelism.
  */
  @chpldoc.nodoc
  record Segment
  {
    type eltType;

    var globalSteal: atomic bool = false;

    var block: unmanaged Block(eltType);

    // private variables
    var o_split: int;
    /* var o_allstolen: bool; */
    var tail: int;

    // shared variables
    var split: atomic int;
    var head: atomic int;
    /* var allstolen: atomic bool; */
    var split_request: atomic bool;
    var nElts_shared: atomic int; // number of elements in the shared space

    // locks (initially unlocked)
    var lock: sync bool = true;
    var lock_n: sync bool = true;
    var lock_block: sync bool = true;

    proc init(type eltType)
    {
      this.eltType = eltType;
      this.block = new unmanaged Block(eltType, distributedBagInitialBlockCap);
    }

    /*
      Returns the size of the private region. This information is computed from the
      tail and split pointers, and since the block is implemented as a  circular
      array, two cases need to be distinguished.
    */
    inline proc nElts_private
    {
      return tail - o_split;
    }

    inline proc nElts
    {
      return nElts_private + nElts_shared.read();
    }

    inline proc isEmpty
    {
      lock_n.readFE();
      var n_shared = nElts_shared.read();
      var n_private = nElts_private;
      lock_n.writeEF(true);
      return (n_shared + n_private) == 0;
    }

    /*
      Insertion operation, only executed by the segment's owner.
    */
    inline proc ref addElement(elt: eltType): bool
    {
      // allocate a larger block with the double capacity.
      if block.isFull {
        if (block.cap == distributedBagMaxBlockCap) then
          return false;
        lock_block.readFE();
        block.cap = min(distributedBagMaxBlockCap, 2*block.cap);
        block.dom = {0..#block.cap};
        lock_block.writeEF(true);
      }

      // we add the element at the tail
      block.pushTail(elt);
      tail += 1;

      // if there is a split request...
      if split_request.read() then split_release();

      /* if o_allstolen {
        lock.readFE(); // block until its full and set locked (empty)
        head.write(tail - 1);
        split.write(tail);
        lock.writeEF(true); // set unlocked (full)
        o_split = tail;
        allstolen.write(false);
        o_allstolen = false;
        if split_request.read() then split_request.write(false);
      }
      else if split_request.read() then split_release(); */

      return true;
    }

    inline proc ref addElements(elts): int
    {
      const size = elts.size;
      var realSize = size;

      // allocate a larger block.
      if (block.tailId + size > block.cap) {
        //TODO: use divceilpos?
        const neededCap = block.cap*2**divCeil(block.tailId + size, block.cap);
        if (neededCap >= distributedBagMaxBlockCap) {
          realSize = distributedBagMaxBlockCap - block.tailId;
        }
        lock_block.readFE();
        block.cap = min(distributedBagMaxBlockCap, neededCap);
        block.dom = {0..#block.cap};
        lock_block.writeEF(true);
      }

      // TODO: find a better way to do the following.
      var c = 0;
      for elt in elts {
        if (c >= realSize) then break;
        block.pushTail(elt);
        c += 1;
      }
      tail += realSize;

      // if there is a split request...
      if split_request.read() then split_release();

      return realSize;
    }

    // TODO: implement 'addElementsPtr'

    /*
      Retrieve operation, only executed by the segment's owner.
    */
    inline proc ref takeElement(): (bool, eltType)
    {

      // if the segment is empty...
      if (nElts_private == 0) {
        var default: eltType;
        return (false, default);
      }

      /* if o_allstolen then {
        var elem = block!.popTail();
        nElems_private.sub(1);
        return (true, elem);
      } */

      // if the private region is empty...
      if (nElts_private == 0) { //(o_split == tail) {
        // if we successfully shring the shared region...
        if split_reacquire() {
          var elt = block.popTail();
          tail -= 1; //?

          return (true, elt);
        }
      }

      // if the private region is not empty...
      var elt = block.popTail();
      tail -= 1;

      // if there is a split request...
      if split_request.read() then split_release();

      return (true, elt);
    }

    // TODO: implement 'takeElements'

    // TODO: implement 'transferElements'

    inline proc simCAS(A: atomic int, B: atomic int, expA: int, expB: int, desA: int, desB: int): bool
    {
      var casA, casB: bool;
      lock.readFE(); // set locked (empty)
      casA = A.compareAndSwap(expA, desA);
      casB = B.compareAndSwap(expB, desB);
      if (casA && casB) {
        lock.writeEF(true); // set unlocked (full)
        return true;
      }
      else {
        if casA then A.write(expA);
        if casB then B.write(expB);
        lock.writeEF(true); // set unlocked (full)
        return false;
      }
      halt("DistributedBag_DFS Internal Error: DEADCODE");
    }

    /*
      Stealing operation, only executed by thieves.
    */
    inline proc ref stealElement(): (bool, eltType)
    {
      var default: eltType;

      // if the shared region becomes empty due to a concurrent operation...
      if (nElts_shared.read() == 0) then return (false, default);

      // Fast exit
      /* if allstolen.read() then return (false, default); */

      lock.readFE(); // set locked (empty)
      var (h, s): (int, int) = (head.read(), split.read());
      lock.writeEF(true); // set unlocked (full)

      // if there are elements to steal...
      if (h < s) {
        // if we successfully moved the pointers...
        if simCAS(head, split, h, s, h+1, s) {
          lock_n.readFE();
          var elt = block.popHead();

          nElts_shared.sub(1);
          lock_n.writeEF(true);

          return (true, elt);
        }
        else {
          return (false, default);
        }
      }

      // set the split request, if not already set...
      if !split_request.read() then split_request.write(true);

      return (false, default);
    }

    /*
      Grow operation that increases the shared space of the deque.
    */
    inline proc ref split_release(): void
    {
      // fast exit
      if (nElts_private <= 1) then return;

      // compute the new split position
      var new_split: int = ((o_split + tail + 1) / 2): int;
      lock.readFE(); // block until its full and set locked (empty)
      split.write(new_split);
      lock.writeEF(true); // set unlocked (full)

      // updates the counters
      lock_n.readFE();
      nElts_shared.add(new_split - o_split);
      lock_n.writeEF(true);

      o_split = new_split;

      // reset split_request
      split_request.write(false);
    }

    /*
      Shrink operation that reduces the shared space of the deque.
    */
    inline proc ref split_reacquire(): bool
    {
      // fast exit
      if (nElts_shared.read() <= 1) then return false;

      lock.readFE(); // block until its full and set locked (empty)
      var (h, s): (int, int) = (head.read(), split.read()); // o_split ?
      lock.writeEF(true); // set unlocked (full)
      if (h != s) {
        var new_split: int = ((h + s) / 2): int;
        lock.readFE(); // block until its full and set locked (empty)
        split.write(new_split);
        lock.writeEF(true); // set unlocked (full)
        lock_n.readFE();
        nElts_shared.sub(new_split - o_split);
        lock_n.writeEF(true);
        o_split = new_split;
        // ADD FENCE
        atomicFence();
        h = head.read();
        if (h != s) {
          if (h > new_split) {
            new_split = ((h + s) / 2): int;
            lock.readFE(); // block until its full and set locked (empty)
            split.write(new_split);
            lock.writeEF(true); // set unlocked (full)
            lock_n.readFE();
            nElts_shared.sub(new_split - o_split);
            lock_n.writeEF(true);
            o_split = new_split;
          }
          return false;
        }
      }
      /* allstolen.write(true);
      o_allstolen = true; */
      return true;
    }
  } // end 'Segment' record

  /*
    A segment block is an unrolled linked list node that holds a contiguous buffer
    of memory. Each segment block size *should* be a power of two, as we increase the
    size of each subsequent unroll block by twice the size. This is so that stealing
    work is faster in that majority of elements are confined to one area.
    It should be noted that the block itself is not parallel-safe, and access must be
    synchronized.
  */
  @chpldoc.nodoc
  class Block
  {
    type eltType;
    var dom: domain(1);
    var elts: [dom] eltType;
    var cap: int; // capacity of the block
    var headId: int; // index of the head element
    var tailId: int; // index of the tail element

    inline proc isFull
    {
      return tailId == cap;
    }

    proc init(type eltType, capacity)
    {
      this.eltType = eltType;
      this.dom = {0..#capacity};
      this.cap = capacity;
    }

    inline proc pushTail(elt: eltType): void
    {
      elts[tailId] = elt;
      tailId += 1;
    }

    inline proc popTail(): eltType
    {
      tailId -= 1;
      return elts[tailId];
    }

    inline proc popHead(): eltType
    {
      var elt = elts[headId];
      headId += 1;
      return elt;
    }
  } // end 'Block' class
} // end module
