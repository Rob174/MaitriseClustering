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
# Friday Feb 04 2022

[1fd3cd6](https://github.com/Rob174/MaitriseClustering/tree/1fd3cd6a02d5e6366256575ed7e9795f3148cbba) 〰️ random_cluster_initialization : wands initialization method but select random clusters to merge at each step
[1fd3cd6](https://github.com/Rob174/MaitriseClustering/tree/1fd3cd6a02d5e6366256575ed7e9795f3148cbba) ⏲️ Debug hmean
# Sunday Feb 06 2022
[12170d3](https://github.com/Rob174/MaitriseClustering/tree/12170d39a3338487e653c728a27259ab9b24d8ec) 🔨 Choice iteration order (CURR, BACK, RANDOM)
[12170d3](https://github.com/Rob174/MaitriseClustering/tree/12170d39a3338487e653c728a27259ab9b24d8ec) ✔️ Visualization callback : Allows to visualize a sequence of solutions
[12170d3](https://github.com/Rob174/MaitriseClustering/tree/12170d39a3338487e653c728a27259ab9b24d8ec) ⏲️ HMeans : cost optimization back to cost=init_cost+variation

# TODO 
1️⃣2️⃣3️⃣4️⃣5️⃣6️⃣7️⃣8️⃣9️⃣🔟 1️⃣ = most urgent 🔟 = can wait
- 1️⃣ implement HMean
  - BI version
  - FI version
    - search cluster from after first cluster CURR
    - search cluster from first cluster in the list of clusters BACK
    - shuffle order of cluster after first choice RAND
  

Suite
1️⃣ Debugguer HMean
2️⃣ Test best et first improvement
3️⃣ Ordre de choix cluster (CURR, BACK  ou RAND) done but not tested
🔟 Si le temps : intégrer visualisation étape de la recherche par étape plotly --> potentiellement très utile pr debuggage complexe