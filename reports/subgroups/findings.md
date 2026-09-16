# Subgroup findings (validation, served policy)

- product=W (n 69700, 1420 fraud): PR-AUC 0.457 (-0.180 vs the grouping)
- card_network=discover (n 936, 120 fraud): block precision 0.67 on 57 blocks
- identity=absent (n 70294, 1460 fraud): PR-AUC 0.457 (-0.180 vs the grouping)
- email_family=anonymous (n 4824, 91 fraud): PR-AUC 0.525 (-0.112 vs the grouping)
- email_family=apple (n 1152, 38 fraud): PR-AUC 0.497 (-0.139 vs the grouping)
- email_family=yahoo (n 15623, 293 fraud): PR-AUC 0.530 (-0.106 vs the grouping)
- device_type=<missing> (n 70694, 1487 fraud): PR-AUC 0.461 (-0.176 vs the grouping)
- amount_band=0-25 (n 6096, 341 fraud): block precision 0.66 on 289 blocks
