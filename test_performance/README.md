=====================================================================================
FINAL COMPARISON REPORT: SPUN vs LIS
=====================================================================================

SCENARIO 1: CLEAR-SKY (Strict Algorithm Comparison)
Definition: Pixels where BOTH Ground Truth and LIS are cloud-free.
-------------------------------------------------------------------------------------
Model           | Recall     | Precis     | F1         | Dice       | MAE        | Pixels    
-------------------------------------------------------------------------------------
SPUN (Clear)    | 0.211      | 0.000      | 0.001      | 0.001      | 19.488     | 334,453   
LIS (Clear)     | 0.439      | 0.562      | 0.493      | 0.493      | 0.049      | 334,453   
-------------------------------------------------------------------------------------

SCENARIO 2: ALL-SKY (Operational Product Comparison)
Definition: All pixels where Ground Truth is valid. LIS Clouds treated as 0 (Miss).
-------------------------------------------------------------------------------------
Model           | Recall     | Precis     | F1         | Dice       | MAE        | Pixels    
-------------------------------------------------------------------------------------
SPUN (All)      | 0.315      | 0.002      | 0.005      | 0.005      | 19.899     | 1,646,808 
LIS (All)       | 0.013      | 0.562      | 0.026      | 0.026      | 0.269      | 1,646,808 
-------------------------------------------------------------------------------------

SCENARIO 3: THE CLOUD GAP (Coverage Analysis)
Total Confirmed Snow Pixels (GT): 3,789
Snow Pixels Detected by LIS:      89  (2.3%)
Snow Pixels Detected by SPUN:     485,013  (12800.6%)
Gain in Snow Detection:           +484,924 pixels
=====================================================================================



we are calculating overall rmse and mae but I also want a snow only rmse and mae because I think this is more informative because I have a highly imblalanced dataset, can you add these?