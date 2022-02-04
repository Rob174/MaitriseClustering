## Progress
✔️ done and tested ; 🔨 done not tested ; ⏲️ in progress ; ⏳ waiting for other scripts to finish ; 🚩 problem ; 🐛 bug ; 〰️ ok does the job but maybe to improve ; 🛑 pause ; 🛰️ release

# Tuesday Feb 01 2022
[ef45671](https://github.com/Rob174/MaitriseClustering/tree/ef4567111a1574f222cbe2243ae28a8e41406d77) Initial version HMean. Creation of the data structures for the algorithm. To be checked. Multiple places to check.
# Wednesday Feb 02 2022
[c9a3581](https://github.com/Rob174/MaitriseClustering/tree/c9a3581df4940583b887fd1b18e0e604f8dd2044) 🔨 Best and first improvement 
[d4bb311](https://github.com/Rob174/MaitriseClustering/tree/d4bb311deeaf93c2dfaa56e3a0390202db8c944f) 🐛🔨 hmean global local search
[d4bb311](https://github.com/Rob174/MaitriseClustering/tree/d4bb311deeaf93c2dfaa56e3a0390202db8c944f) ✔️ KMeans+ 
[d4bb311](https://github.com/Rob174/MaitriseClustering/tree/d4bb311deeaf93c2dfaa56e3a0390202db8c944f) ✔️ Simple test 1
✔️🚩 2 Initialization methods  as in First vs best
[a6c1ea0](https://github.com/Rob174/MaitriseClustering/tree/a6c1ea021bea534bb405618572a4d0d8c2b6499d) ✔️ Cost of the initial solution
〰️ random_cluster_initialization : wands initialization method but select random clusters to merge at each step
⏲️ Debug hmean
# TODO 
1️⃣2️⃣3️⃣4️⃣5️⃣6️⃣7️⃣8️⃣9️⃣🔟 1️⃣ = most urgent 🔟 = can wait
- 1️⃣ implement HMean
  - BI version
  - FI version
    - search cluster from after first cluster CURR
    - search cluster from first cluster in the list of clusters BACK
    - shuffle order of cluster after first choice RAND
  
