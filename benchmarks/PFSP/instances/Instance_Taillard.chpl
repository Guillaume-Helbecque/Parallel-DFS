use CTypes;
use Instance;

require "../c_sources/c_taillard.c", "../c_headers/c_taillard.h";

extern proc taillard_get_nb_jobs(const inst_id: c_int): c_int;
extern proc taillard_get_nb_machines(const inst_id: c_int): c_int;
extern proc taillard_get_processing_times(data: c_ptr(c_int), const id: c_int): void;

class Instance_Taillard : Instance
{
  var id: c_int;

  proc init(const fileName: string)
  {
    this.name = fileName;
    this.id = name[2..]: c_int;

    if (this.id < 1 || 120 < this.id) then halt("Error - Unknown Taillard instance");
  }

  override proc get_nb_jobs(): c_int
  {
    return taillard_get_nb_jobs(this.id);
  }

  override proc get_nb_machines(): c_int
  {
    return taillard_get_nb_machines(this.id);
  }

  override proc get_data(d: c_ptr(c_int))
  {
    taillard_get_processing_times(d, this.id);
  }

  override proc get_ub()
  {
    var optimal: [1..120] int =  [1278, 1359, 1081, 1293, 1235, 1195, 1234, 1206, 1230, 1108,            // 20x5
                                  1582, 1659, 1496, 1377, 1419, 1397, 1484, 1538, 1593, 1591,            // 20x10
                                  2297, 2099, 2326, 2223, 2291, 2226, 2273, 2200, 2237, 2178,            // 20x20

                                  2724, 2834, 2621, 2751, 2863, 2829, 2725, 2683, 2552, 2782,            // 50x5
                                  2991, 2867, 2839, 3063, 2976, 3006, 3093, 3037, 2897, 3065,            // 50x10
                                  3850, 3704, 3640, 3723, 3611, 3681, 3704, 3691, 3743, 3756,            // 50x20

                                  5493, 5268, 5175, 5014, 5250, 5135, 5246, 5094, 5448, 5322,            // 100x5
                                  5770, 5349, 5676, 5781, 5467, 5303, 5595, 5617, 5871, 5845,            // 100x10
                                  6202, 6183, 6271, 6269, 6314, 6364, 6268, 6401, 6275, 6434,            // 100x20

                                  10862, 10480, 10922, 10889, 10524, 10329, 10854, 10730, 10438, 10675,  // 200x10
                                  11195, 11203, 11281, 11275, 11259, 11176, 11360, 11334, 11192, 11284,  // 200x20

                                  26040, 26520, 26371, 26456, 26334, 26477, 26389, 26560, 26005, 26457]; // 500x20

    return optimal[this.id];
  }
}
