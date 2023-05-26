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
  use BlockDist;
  private use CTypes;

  use Random;
  use Time;
  use List;

  /*
    The phases for the remove operation. A remove operation is composed of multiple
    phases, where it make a full pass searching for ideal conditions, then less-than-ideal
    conditions; this is required to ensure maximized parallelism at all times, and
    critical to good performance, especially when a node is oversubscribed.
  */
  private param REMOVE_SIMPLE         = 1;
  private param REMOVE_LOCAL_STEAL    = 2;
  private param REMOVE_GLOBAL_STEAL   = 3;

  private param REMOVE_SUCCESS   =  1;
  private param REMOVE_FAST_EXIT =  0;
  private param REMOVE_FAIL      = -1;

  /*
    The initial capacity of a block. Each expanded block's capacity is double the
    old one, allowing for better locality for when there are larger numbers of
    elements. The better the locality, the better raw performance and easier it
    is to redistribute work.
  */
  config const distributedBagInitialBlockCap: int = 1024;
  /*
    To prevent stealing too many elements (horizontally) from another node's segment
    (hence creating an artificial load imbalance), if the other node's segment has
    less than a certain threshold (see: const:`distributedBagWorkStealingMemCap`)
    but above another threshold (see: const:`distributedBagWorkStealingMinElems`),
    we steal a percentage of their elements, leaving them with majority of their
    elements. This way, the amount the other segment loses is proportional to how
    much it owns, ensuring a balance.
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
  config const distributedBagWorkStealingMinElems: int = 1;
  /*
    The maximum capacity of a block. This is crucial to ensure memory usage does
    not rapidly grow out of control.
  */
  config const distributedBagMaxBlockCap: int = 1024 * 1024;

  /*
    Reference counter for DistributedBag_DFS
  */
  pragma "no doc"
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
    A parallel-safe distributed multi-pool implementation that scales in terms of
    nodes, processors per node (PPN), and workload; The more PPN, the more segments
    we allocate to increase raw parallelism, and the larger the workload the better
    locality (see: const:`distributedBagInitialBlockCap`). This data structure is
    unordered and employs its own work-stealing algorithm, and provides a means to
    obtain a privatized instance of the data structure for maximized performance.
  */
  pragma "always RVF"
  record DistBag_DFS
  {
    type eltType;

    // This is unused, and merely for documentation purposes. See '_value'.
    /*
      The implementation of the Bag is forwarded. See: class:`DistributedBagImpl`
      for documentation.
    */
    var _impl: unmanaged DistributedBagImpl(eltType)?;

    // Privatized id
    pragma "no doc"
    var _pid: int = -1;

    // Reference Counting
    pragma "no doc"
    var _rc: shared DistributedBagRC(eltType);

    pragma "no doc"
    proc init(type eltType, targetLocales = Locales)
    {
      this.eltType = eltType;
      this._pid = (new unmanaged DistributedBagImpl(eltType, targetLocales = targetLocales)).pid;
      this._rc = new shared DistributedBagRC(eltType, _pid = _pid);
    }

    pragma "no doc"
    inline proc _value
    {
      if (_pid == -1) then halt("DistBag_DFS is uninitialized.");
      return chpl_getPrivatizedCopy(unmanaged DistributedBagImpl(eltType), _pid);
    }

    pragma "no doc"
    proc readThis(f) throws {
      compilerError("Reading a DistBag_DFS is not supported");
    }

    // Write the contents of DistBag_DFS to a channel.
    pragma "no doc"
    proc writeThis(ch) throws {
      ch.write("[");
      var size = this.getSize();
      for (i, iteration) in zip(this, 0..<size) {
        ch.write(i);
        if (iteration < size-1) then ch.write(", ");
      }
      ch.write("]");
    }

    forwarding _value;
  }

  class DistributedBagImpl : CollectionImpl
  {
    pragma "no doc"
    var targetLocDom: domain(1);

    /*
      The locales to allocate bags for and load balance across.
    */
    var targetLocales: [targetLocDom] locale;
    pragma "no doc"
    var pid: int = -1;

    // Node-local fields below. These fields are specific to the privatized instance.
    // To access them from another node, make sure you use 'getPrivatizedThis'
    pragma "no doc"
    var bag: unmanaged Bag(eltType)?;

    proc init(type eltType, targetLocales: [?targetLocDom] locale = Locales)
    {
      super.init(eltType);

      this.targetLocDom  = targetLocDom;
      this.targetLocales = targetLocales;

      complete();

      this.pid = _newPrivatizedClass(this);
      this.bag = new unmanaged Bag(eltType, this);
    }

    pragma "no doc"
    proc init(other, pid, type eltType = other.eltType)
    {
      super.init(eltType);

      this.targetLocDom  = other.targetLocDom;
      this.targetLocales = other.targetLocales;
      this.pid           = pid;

      complete();

      this.bag = new unmanaged Bag(eltType, this);
    }

    pragma "no doc"
    proc deinit()
    {
      delete bag;
    }

    pragma "no doc"
    proc dsiPrivatize(pid)
    {
      return new unmanaged DistributedBagImpl(this, pid);
    }

    pragma "no doc"
    proc dsiGetPrivatizeData()
    {
      return pid;
    }

    pragma "no doc"
    inline proc getPrivatizedThis
    {
      return chpl_getPrivatizedCopy(this.type, pid);
    }

    pragma "no doc"
    iter targetLocalesNotHere()
    {
      foreach loc in targetLocales {
        if (loc != here) then yield loc;
      }
    }

    /*
      Insert an element to this node's bag. The ordering is not guaranteed to be
      preserved.
    */
    pragma "no doc"
    override proc add(elt: eltType): bool
    {
      return bag!.add(elt);
    }

    /*
      Insert an element in the calling task's segment of this node's bag.
    */
    proc add(elt: eltType, tid: int): bool
    {
      return bag!.add(elt, tid);
    }

    /*
      Insert elements in bulk in the calling task's segment of this node's bag.
      If the node's bag rejects an element, we cease to offer more. We return the
      number of elements successfully added to the data structure.
    */
    proc addBulk(elts, taskId: int): int
    {
      var success: int;
      for elt in elts {
        if !add(elt, taskId) then break;

        success += 1;
      }

      return success;
    }

    /*
      Remove an element from this node's bag. The order in which elements are removed
      are not guaranteed to be the same order it has been inserted. If this node's
      bag is empty, it will attempt to steal elements from bags of other nodes.
    */
    pragma "no doc"
    override proc remove(): (bool, eltType)
    {
      return bag!.remove();
    }

    /*
      Remove an element from the calling task's segment of this node's bag.
      If the task's segment is empty, it will attempt to steal an element from
      the segment of another task.
    */
    proc remove(tid: int): (int, eltType)
    {
      return bag!.remove(tid);
    }

    proc removeBulk_(n: int, tid: int): (bool, [0..#n] eltType)
    {
      return bag!.removeBulk_(n, tid);
    }

    /*
      Obtain the number of elements held in all bags across all nodes. This method
      is best-effort and can be non-deterministic for concurrent updates across nodes,
      and may miss elements or even count duplicates resulting from any concurrent
      insertion or removal operations.
    */
    override proc getSize(): int
    {
      var sz: atomic int;
      coforall loc in targetLocales do on loc {
        var instance = getPrivatizedThis;
        forall segmentIdx in 0..#here.maxTaskPar {
          sz.add(instance.bag!.segments[segmentIdx].nElems.read():int);
        }
      }

      return sz.read();
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
      var localThis = getPrivatizedThis;
      coforall loc in localThis.targetLocales do on loc {
        var instance = getPrivatizedThis;
        forall segmentIdx in 0..#here.maxTaskPar {
          ref segment = instance.bag!.segments[segmentIdx];

          segment.lock_block$.readFE();

          delete segment.block;
          segment.block = nil;
          segment.o_split = 0; segment.o_allstolen = false; segment.tail = 0;
          segment.split.write(0); segment.allstolen.write(false); segment.head.write(0);
          segment.split_request.write(false); segment.nElems_shared.write(0);
          segment.globalSteal.write(false);

          segment.lock_block$.writeEF(true);
        }
      }
    }

    /*
      Triggers a more static approach to load balancing, fairly redistributing all
      elements fairly for bags across nodes. The result will result in all segments
      having roughly the same amount of elements.
      .. note::
      This method is very heavy-weight in that it should not be called too
      often. Dynamic work stealing handles cases where there is a relatively fair
      distribution across majority of nodes, but this should be called when you have
      a severe imbalance, or when you have a smaller number of elements to balance.
      Furthermore, while this operation is parallel-safe, it should be called from at
      most one task.
    */

    // UNUSED (balance)
    /* proc balance(): void
    {
      var localThis = getPrivatizedThis;
      // Phase 1: Acquire all locks from first node and segment to last
      // node and segment (our global locking order...)
      for loc in localThis.targetLocales do on loc {
        var instance = getPrivatizedThis;
        for segmentIdx in 0..#here.maxTaskPar {
          ref segment = instance.bag!.segments[segmentIdx];
          segment.acquire(STATUS_BALANCE);
        }
      }

      // Phase 2: Concurrently redistribute elements from segments which contain
      // more than the computed average.
      coforall segmentIdx in 0..#here.maxTaskPar {
        var nSegmentElems: [0..#localThis.targetLocales.size] int;
        var locIdx = 0;
        for loc in localThis.targetLocales do on loc {
          var nElems = getPrivatizedThis.bag!.segments[segmentIdx].nElems:int;
          nSegmentElems[locIdx] = nElems;
          locIdx += 1;
        }

        // Find the average and the excess. The excess is calculated as the amount
        // of elements a segment has over the average, which is used to calculate
        // the buffer size for each segment.
        var total = (+ reduce nSegmentElems);
        var avg = total / locIdx;
        var excess: int;
        for nElems in nSegmentElems {
          if (nElems > avg) then excess += nElems - avg;
        }

        // Allocate buffer, which holds the 'excess' elements for redistribution.
        // Then fill it.
        var buffer = c_malloc(eltType, excess);
        var bufferOffset = 0;
        for loc in localThis.targetLocales do on loc {
          var average = avg;
          ref segment = getPrivatizedThis.bag!.segments[segmentIdx];
          var nElems = segment.nElems:int;
          if (nElems > average) {
            var nTransfer = nElems - average;
            var tmpBuffer = buffer + bufferOffset;
            segment.transferElements(tmpBuffer, nTransfer, buffer.locale.id);
            bufferOffset += nTransfer;
          }
        }

        // With the excess elements, redistribute it...
        bufferOffset = 0;
        for loc in localThis.targetLocales do on loc {
          var average = avg;
          ref segment = getPrivatizedThis.bag!.segments[segmentIdx];
          var nElems = segment.nElems:int;
          if (average > nElems) {
            var give = average - nElems;
            var arr: [1..give] eltType;
            on bufferOffset {
              var tmpBuffer = buffer;
              for i in 1..give {
                arr[i] = tmpBuffer[bufferOffset];
                bufferOffset += 1;
              }
            }
            for i in 1..give {
              segment.addElement(arr[i]);
            }
          }
        }

        // Lastly, if there are items left over, just add them to our locale's segment.
        if (excess > bufferOffset) {
          ref segment = localThis.bag!.segments[segmentIdx];
          var nLeftOvers = excess - bufferOffset;
          var tmpBuffer = buffer + bufferOffset;
          segment.addElementsPtr(tmpBuffer, nLeftOvers, buffer.locale.id);
        }

        c_free(buffer);
      }

      // Phase 3: Release all locks from first node and segment to last node and segment.
      for loc in localThis.targetLocales do on loc {
        var instance = getPrivatizedThis;
        for segmentIdx in 0..#here.maxTaskPar {
          ref segment = instance.bag!.segments[segmentIdx];
          segment.releaseStatus();
        }
      }
    } */

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

    // UNUSED (these)
    /* override iter these(): eltType
    {
      for loc in targetLocales {
        for segmentIdx in 0..#here.maxTaskPar {
          // The size of the snapshot is only known once we have the lock.
          var dom : domain(1) = {0..-1};
          var buffer : [dom] eltType;

          on loc {
            ref segment = getPrivatizedThis.bag!.segments[segmentIdx];

            if segment.acquireIfNonEmpty(STATUS_LOOKUP) {
              dom = {0..#segment.nElems.read():int};
              var block = segment.headBlock;
              var idx = 0;
              while (block != nil) {
                for i in 0..#block!.size {
                  buffer[idx] = block!.elems[i];
                  idx += 1;
                }
                block = block!.next;
              }
              segment.releaseStatus();
            }
          }

          // Process this chunk if we have one...
          foreach elem in buffer {
            yield elem;
          }
        }
      }
    } */

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
                halt("DistributedBag_DFS Internal Error: Snapshot attempt with bufferSz(", bufferSz, ") with offset bufferOffset(", bufferOffset + block!.size, ")");
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
  }

  /*
    We maintain a multi-pool 'bag' per node. Each bag keeps a handle to it's parent,
    which is required for work stealing.
  */
  pragma "no doc"
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
    }

    proc deinit()
    {
      forall segment in segments {
        delete segment.block;
      }
    }

    /*
      This iterator is intented to select victim(s) in work-stealing strategies,
      according to the specified policy. By default, the 'rand' strategy is chosen and
      the calling thread/locale cannot be chosen. We can specify how many tries we want,
      by default, only 1 is performed.
    */
    iter victim(const N: int, const callerId: int, const mode: string = "rand", const tries: int = 1): int
    {
      var count: int;

      select mode {
        // In the 'ring' strategy, victims are selected in a round-robin fashion.
        when "ring" {
          var id = (callerId + 1) % N;

          while ((count < N-1) && (count < tries)){
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

          while ((count < N-1) && (count < tries)){
            if (victims[id] != callerId){
              yield victims[id];
              count += 1;
            }
            id += 1;
          }
        }
        otherwise halt("DistributedBag_DFS internal error: Wrong victim choice policy");
      }
    }

    proc add(elt: eltType): bool
    {
      halt("DistributedBag_DFS Internal Error: DEADCODE.");
    }

    /*
      Insertion operation.
    */
    proc add(elt: eltType, const taskId: int): bool
    {
      segments[taskId].addElement(elt);

      return true;
    }

    proc remove(): (bool, eltType)
    {
      halt("DistributedBag_DFS Internal Error: DEADCODE.");
    }

    // This procedure is only callable by the owner thread
    proc removeBulk_(const n: int, const taskId: int)//: (bool, [0..] eltType)
    {
      ref segment = segments[taskId];
      var default: [0..-1] eltType;

      // if the private region contains enough elements to be removed...
      if (segment.tail - segment.o_split >= n) {
        var elts: [0..#n] eltType;
        for elt in elts do elt = segment.takeElement();
        return (true, elts);
      }

      return (false, default);
    }

    /*
      Retrieval operation that succeeds when one of the three successive cases
      succeeds:
        - SIMPLE CASE: the caller try to remove an element from its segment.
        - LOCAL STEAL CASE: the caller try to steal another segment of its bag
            instance.
        - GLOBAL STEAL CASE: the caller try to steal another segment of another
            bag instance.
      The operation fails if all cases failed.
    */
    proc remove(const taskId: int): (int, eltType)
    {
      var phase = REMOVE_SIMPLE;

      while true {
        select phase {
          /*
            SIMPLE CASE:
            If the private region is not empty, we take an element.
            If the private region is empty, we try to reacquire elements by
            shrinking the public region:
              - if success: we take an element in the (new) private region;
              - if fail: LOCAL STEAL CASE.
          */
          when REMOVE_SIMPLE {
            ref segment = segments[taskId];

            // if the private region contains at least one element to be removed...
            if (segment.tail > segment.o_split) {
              if segment.split_request.read() then segment.split_release();
              return (REMOVE_SUCCESS, segment.takeElement());
            }
            else if segment.split_reacquire() {
              return (REMOVE_SUCCESS, segment.takeElement());
            }
            else {
              phase = REMOVE_LOCAL_STEAL;
            }
          }

          /*
            LOCAL STEAL: intra-node work stealing
            It seems that segment 'taskId' is empty so we try to steal another one.
            The victim selection is set in the 'victim' iterator, and defaults to random.
            The work stealing fails when all segments don't satisfied the condition to be
            a victim, or when a shared region becomes empty due to a concurrent operation.
          */
          when REMOVE_LOCAL_STEAL {
            var default: eltType;
            var splitreq: bool = false;

            // fast exit if: (1) a segment of our bag instance is doing a global steal
            // (2) a global steal is performed on our segment (priority is given).
            // if globalStealInProgress.read() then return (REMOVE_FAST_EXIT, default);

            if globalStealInProgress.read() {
              return (REMOVE_FAST_EXIT, default);
            }

            segments[taskId].nSteal1 += 1;
            segments[taskId].timer1.start();

            // selection of the victim segment
            for idx in victim(here.maxTaskPar, taskId, "rand", here.maxTaskPar) {
              ref targetSegment = segments[idx];

              if !targetSegment.globalSteal.read() {
                targetSegment.lock_block$.readFE();
                // if the shared region contains enough elements to be stolen...
                if (distributedBagWorkStealingMinElems <= targetSegment.nElems_shared.read()) {
                  // attempt to steal an element
                  var (hasElem, elem): (bool, eltType) = targetSegment.steal();

                  // if the steal succeeds, we return, otherwise we continue
                  if hasElem {
                    targetSegment.lock_block$.writeEF(true);
                    segments[taskId].timer1.stop();
                    segments[taskId].nSSteal1 += 1;
                    return (REMOVE_SUCCESS, elem);
                  }
                }
                // otherwise, if the private region has elements, we request for a split shifting
                else if (targetSegment.nElems_private > 1) {
                  splitreq = true;
                  targetSegment.split_request.write(true);
                }
                targetSegment.lock_block$.writeEF(true);
              }
            }

            segments[taskId].timer1.stop();

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
            var default: eltType;
            //var (hasStolen, stolenElt): (bool, eltType) = (false, default);

            // fast exit for single-node execution
            if (numLocales == 1) then return (REMOVE_FAIL, default);

            // "Lock" the global steal operation
            if !globalStealInProgress.compareAndSwap(false, true) {
              return (REMOVE_FAST_EXIT, default);
            }

            const parentPid = parentHandle.pid;
            var stolenElts: list(eltType);
            var timer, subtimer: stopwatch;

            segments[taskId].nSteal2 += 1;
            segments[taskId].timer2.start();

            timer.start();

            // selection of the victim locale
            for idx in victim(numLocales, here.id, "rand", 1) { //numLocales-1) {
              on Locales[idx] {
                var targetBag = chpl_getPrivatizedCopy(parentHandle.type, parentPid).bag;
                // selection of the victim segment
                for seg in victim(here.maxTaskPar, taskId, "rand", here.maxTaskPar) { //0..#here.maxTaskPar {
                  ref targetSegment = targetBag!.segments[seg];

                  targetSegment.globalSteal.write(true);

                  //var sharedElts: int = targetSegment.nElems_shared.read();
                  // if the shared region contains enough elements to be stolen...
                  targetSegment.lock_block$.readFE();
                  if (2 <= targetSegment.nElems_shared.read()) {
                    //for i in 0..#(targetSegment.nElems_shared.read()/2):int {
                      // attempt to steal an element
                      var (hasElem, elem): (bool, eltType) = targetSegment.steal();

                      subtimer.start();
                      // if the steal succeeds...
                      if hasElem {
                        stolenElts.insert(0, elem);
                      }
                      subtimer.stop();
                  //  }
                  }
                  // otherwise, if the private region has elements, we request for a split shifting
                  else if (targetSegment.nElems_private >= 2) {
                    targetSegment.split_request.write(true);
                  }

                  targetSegment.lock_block$.writeEF(true);
                  targetSegment.globalSteal.write(false);
                }
              }
            }

            // if the global steal fails...
            if (stolenElts.size == 0) {
              // "Unlock" the global steal operation
              globalStealInProgress.write(false);
              timer.stop();
              segments[taskId].timer2.stop();
              return (REMOVE_FAIL, default);
            }
            else {
              /* writeln(stolenElts.size); */
              segments[taskId].addElements(stolenElts);
              //segments[taskId].split.add((3*stolenElts.size/4):int);
              segments[taskId].nSSteal2 += 1;

              // "Unlock" the global steal operation
              globalStealInProgress.write(false);
              timer.stop();
              segments[taskId].timer2.stop();
              return (REMOVE_SUCCESS, segments[taskId].takeElement());
            }
          }

          otherwise do halt("DistributedBag_DFS Internal Error: Invalid phase #", phase);
        }
        chpl_task_yield();
      }

      halt("DistributedBag_DFS Internal Error: DEADCODE.");
    }
  } // end 'Bag' class

  /*
    A Segment is, in and of itself an unrolled linked list. We maintain one per core
    to ensure maximum parallelism.
  */
  pragma "no doc"
  record Segment
  {
    type eltType;

    var globalSteal: atomic bool = false;

    var block: unmanaged Block(eltType)?;

    // private variables
    var o_split: int;
    var o_allstolen: bool;
    var tail: int;

    // shared variables
    var split: atomic int;
    var head: atomic int;
    var allstolen: atomic bool;
    var split_request: atomic bool;
    var nElems_shared: atomic int; // number of elements in the shared space

    // for profiling
    var nSteal1: int;
    var nSSteal1: int;
    var nSteal2: int;
    var nSSteal2: int;

    var timer1, timer2: stopwatch;

    // locks (initially unlocked)
    var lock$: sync bool = true;
    var lock_n$: sync bool = true;
    var lock_block$: sync bool = true;

    /*
      Returns the size of the private region. This information is computed from the
      tail and split pointers.
    */
    inline proc nElems_private
    {
      return tail - o_split;
    }

    inline proc nElems
    {
      return nElems_private + nElems_shared.read();
    }

    inline proc isEmpty
    {
      lock_n$.readFE();
      var n_shared = nElems_shared.read();
      var n_private = nElems_private;
      lock_n$.writeEF(true);
      return (n_shared + n_private) == 0;
    }

    /* inline proc transferElements(destPtr, n, locId = here.id)
    {
      var destOffset = 0;
      var srcOffset = 0;
      while (destOffset < n) {
        if ((block == nil) || isEmpty) {
          halt(here, ": DistributedBag_DFS Internal Error: Attempted transfer ", n, " elements to ", locId, " but failed... destOffset=", destOffset);
        }

        var len = block!.size;
        var need = n - destOffset;
        // If the amount in this block is greater than what is left to transfer, we
        // cannot begin transferring at the beginning, so we set our offset from the end.
        if (len > need) {
          srcOffset = len - need;
          block!.size = srcOffset;
          __primitive("chpl_comm_array_put", block!.elems[srcOffset], locId, destPtr[destOffset], need);
          destOffset += need;
        } else {
          srcOffset = 0;
          block!.size = 0;
          __primitive("chpl_comm_array_put", block!.elems[srcOffset], locId, destPtr[destOffset], len);
          destOffset += len;
        }
      }

      nElems.sub(n:int);
    } */

    // UNUSED (addElementsPtr) Only in balance() and old remove()
    /* proc addElementsPtr(ptr, n, locId = here.id)
    {
      var offset = 0;
      while (offset < n) {

        // Empty? Create a new one of initial size
        if (block == nil) {
          block = new unmanaged Block(eltType, distributedBagInitialBlockCap);
        }

        // Full? Create a new one double the previous size
         if block!.isFull {
          block!.next = new unmanaged Block(eltType, min(distributedBagMaxBlockCap, block!.cap * 2));
          block = block!.next;
        }

        var nLeft = n - offset;
        var nSpace = block!.cap - block!.size;
        var nFill = min(nLeft, nSpace);
        __primitive("chpl_comm_array_get", block!.elems[block!.size], locId, ptr[offset], nFill);
        block!.size += nFill;
        offset += nFill;
      }

      nElems.add(n:int);
    } */

    // UNUSED (takeElements)
    /* inline proc takeElements(n, side: string)
    {
      var iterations = n:int;
      var arr: [{0..#n : int}] eltType;
      var arrIdx = 0;

      for 1..n : int {
        if isEmpty then halt("DistributedBag_DFS Internal Error: Attempted to take ", n, " elements when insufficient");
        if headBlock!.isEmpty then halt("DistributedBag_DFS Internal Error: Iterating over ", n, " elements with headBlock empty but nElems is ", nElems.read());

        arr[arrIdx] = headBlock.pop();
        arrIdx += 1;
        nElems.sub(1);

        // Fix list if we consumed last one...
        if headBlock!.isEmpty {
          var tmp = headBlock;
          headBlock = headBlock!.next;
          delete tmp;

          if (headBlock == nil) then tailBlock = nil;
        }
      }

      return arr;
    } */

    inline proc simCAS(A: atomic int, B: atomic int, expA: int, expB: int, desA: int, desB: int): bool
    {
      var casA, casB: bool;
      lock$.readFE(); // set locked (empty)
      casA = A.compareAndSwap(expA, desA);
      casB = B.compareAndSwap(expB, desB);
      if (casA && casB) {
        lock$.writeEF(true); // set unlocked (full)
        return true;
      }
      else {
        if casA then A.write(expA);
        if casB then B.write(expB);
        lock$.writeEF(true); // set unlocked (full)
        return false;
      }
      halt("DistributedBag_DFS Internal Error: DEADCODE");
    }

    /*
      Stealing operation, only executed by thieves.
    */
    inline proc steal(): (bool, eltType)
    {
      var default: eltType;

      // if the shared region becomes empty due to a concurrent operation...
      if (nElems_shared.read() == 0) then return (false, default);

      // Fast exit
      /* if allstolen.read() then return (false, default); */

      lock$.readFE(); // set locked (empty)
      var (h, s): (int, int) = (head.read(), split.read());
      lock$.writeEF(true); // set unlocked (full)

      // if there are elements to steal...
      if (h < s) {
        // if we successfully moved the pointers...
        if simCAS(head, split, h, s, h+1, s) {
          lock_n$.readFE();
          var elem = block!.popHead();

          nElems_shared.sub(1);
          lock_n$.writeEF(true);

          return (true, elem);
        }
        else {
          return (false, default);
        }
      }

      // set the split request, if not already set...
      if !split_request.read() then split_request.write(true);

      return (false, default);
    }

    /* // We can take element at the tail or head block, according to the 'side' argument.
    inline proc takeElement(side: string)
    {
      // If the segment is empty
      if isEmpty {
        var default: eltType;
        return (false, default);
      }

      if block!.isEmpty then halt("DistributedBag_DFS Internal Error: Iterating over 1 element with headBlock empty but nElems is ", nElems.read());

      if (side == 't') { // remove at the 't'ail
        var elem = block!.popTail();
        nElems.sub(1);

        if block!.isEmpty { // if block is now empty
          delete block;
          block = nil;
        }

        return (true, elem);
      }
      else if (side == 'h') { // remove at the 'h'ead
        var elem = block!.popHead();
        nElems.sub(1);

        if block!.isEmpty { // if block is now empty
          delete block;
          block = nil;
        }

        return (true, elem);
      }
      else halt("DistributedBag_DFS Internal Error: Wrong 'side' choice in takeElement().");
    } */

    /*
      Retrieve operation, only executed by the segment's owner.
    */
    inline proc takeElement(): eltType
    {
      /* if o_allstolen then {
        var elem = block!.popTail();
        nElems_private.sub(1);

        return (true, elem);
      } */

      var elt = block!.popTail();
      tail -= 1;

      return elt;
    }

    /*
      Insertion operation, only executed by the segment's owner.
    */
    inline proc addElement(elt: eltType)
    {
      // if the block is not already initialized...
      if (block == nil) then block = new unmanaged Block(eltType, distributedBagInitialBlockCap);

      // allocate a larger block if the current one is full
      if block!.isFull {
        var nblock = new unmanaged Block(eltType, min(distributedBagMaxBlockCap, block!.cap * 2));
        lock_block$.readFE();
        nblock!.headIdx = block!.headIdx;
        nblock!.tailIdx = block!.tailIdx;
        for i in 0..#block!.cap {
          nblock!.elems[i] = block!.elems[i];
        }
        delete block;
        block = nblock;
        lock_block$.writeEF(true);
      }

      // we add the element at the tail
      block!.pushTail(elt);
      tail += 1;

      // if there is a split request...
      if split_request.read() then split_release();

      /* if o_allstolen {
        lock$.readFE(); // block until its full and set locked (empty)
        head.write(tail - 1);
        split.write(tail);
        lock$.writeEF(true); // set unlocked (full)
        o_split = tail;
        allstolen.write(false);
        o_allstolen = false;
        if split_request.read() then split_request.write(false);
      }
      else if split_request.read() then split_release(); */
    }

    inline proc addElements(elts)
    {
      for elt in elts do addElement(elt);
    }

    /*
      Grow operation that increases the shared space of the deque.
    */
    inline proc split_release(): void
    {
      // fast exit
      if (nElems_private <= 1) then return;

      // compute the new split position
      var new_split: int = ((o_split + tail + 1) / 2): int;
      lock$.readFE(); // block until its full and set locked (empty)
      split.write(new_split);
      lock$.writeEF(true); // set unlocked (full)

      // updates the counters
      lock_n$.readFE();
      nElems_shared.add(new_split - o_split);
      lock_n$.writeEF(true);

      o_split = new_split;

      // reset split_request
      split_request.write(false);
    }

    /*
      Shrink operation that reduces the shared space of the deque.
    */
    inline proc shrink_shared(): bool
    {
      // fast exit
      if (nElems_shared.read() <= 1) then return false;

      lock$.readFE(); // block until its full and set locked (empty)
      var (h, s): (int, int) = (head.read(), split.read()); // o_split ?
      lock$.writeEF(true); // set unlocked (full)
      if (h != s) {
        var new_split: int = ((h + s) / 2): int;
        lock$.readFE(); // block until its full and set locked (empty)
        split.write(new_split);
        lock$.writeEF(true); // set unlocked (full)
        lock_n$.readFE();
        nElems_shared.sub(new_split - o_split);
        lock_n$.writeEF(true);
        o_split = new_split;
        // ADD FENCE
        atomicFence();
        h = head.read();
        if (h != s) {
          if (h > new_split) {
            new_split = ((h + s) / 2): int;
            lock$.readFE(); // block until its full and set locked (empty)
            split.write(new_split);
            lock$.writeEF(true); // set unlocked (full)
            lock_n$.readFE();
            nElems_shared.sub(new_split - o_split);
            lock_n$.writeEF(true);
            o_split = new_split;
          }
          return false;
        }
      }
      allstolen.write(true);
      o_allstolen = true;
      return true;
    }

    /*
      Shrink operation that reduces the shared space of the deque.
    */
    proc split_reacquire(): bool
    {
      // fast exit
      if (nElems_shared.read() <= 1) then return false;

      lock$.readFE(); // block until its full and set locked (empty)
      var h: int = head.read();

      if (h < o_split) {
//        var new_split: int = o_split - 1;//((h + o_split) / 2): int;
        split.sub(1);//write(new_split);
        lock$.writeEF(true); // set unlocked (full)
        lock_n$.readFE();
        nElems_shared.sub(1);//new_split - o_split);
        lock_n$.writeEF(true);
        o_split -= 1;//new_split;
        return true;
      }
      else {
        lock$.writeEF(true); // set unlocked (full)
        return false;
      }
    }
  } // end Segment record

  /*
    A segment block is an unrolled linked list node that holds a contiguous buffer
    of memory. Each segment block size *should* be a power of two, as we increase the
    size of each subsequent unroll block by twice the size. This is so that stealing
    work is faster in that majority of elements are confined to one area.
    It should be noted that the block itself is not parallel-safe, and access must be
    synchronized.
  */
  pragma "no doc"
  class Block
  {
    type eltType;
    var elems: c_ptr(eltType); // contiguous memory containing all elements
    // TODO: test with elems: cap * eltType

    var cap: int; // capacity of the block
    var headIdx: int; // index of the head element
    var tailIdx: int; // index of the tail element

    /* inline proc isEmpty
    {
      return headIdx == tailIdx;
      return size == 0;
    } */

    inline proc isFull
    {
      return tailIdx == cap;
    }

    // ISSUE: Cannot insert Chapel array due to "c_malloc".
    proc init(type eltType, capacity)
    {
      /* if (capacity == 0) then halt("DistributedBag_DFS Internal Error: Capacity is 0."); */
      this.eltType = eltType;
      this.elems = c_malloc(eltType, capacity);
      this.cap = capacity;
    }

    // UNUSED (init) I think
    /* proc init(type eltType, ptr, capacity)
    {
      this.eltType = eltType;
      this.elems = ptr;
      this.cap = capacity;
      this.size = cap;
    } */

    proc deinit()
    {
      c_free(elems);
    }

    inline proc pushTail(elt: eltType): void
    {
      /* if (elems == nil) then halt("DistributedBag_DFS Internal Error in 'pushTail': 'elems' is nil."); */
      /* if isFull then halt("DistributedBag_DFS Internal Error in 'pushTail': Block is Full."); */

      elems[tailIdx] = elt;
      tailIdx += 1;

      return;
    }

    inline proc popTail(): eltType
    {
      /* if (elems == nil) then halt("DistributedBag_DFS Internal Error in 'popTail': 'elems' is nil."); */
      /* if isEmpty then halt("DistributedBag_DFS Internal Error in 'popTail': Block is Empty."); */

      tailIdx -= 1;

      return elems[tailIdx];
    }

    inline proc popHead(): eltType
    {
      /* if (elems == nil) then halt("DistributedBag_DFS Internal Error in 'popHead': 'elems' is nil."); */
      /* if isEmpty then halt("DistributedBag_DFS Internal Error in 'popHead': Block is Empty."); */

      var elt = elems[headIdx];
      headIdx += 1;

      return elt;
    }
  } // end 'Block' class

} // end module
