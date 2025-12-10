# Kernel methods through the roof: handling billions of points efficiently

## Progress Notes

### Meeting 19/11

Repartition des tâches : 
- Chaqun lis les maths +++
- Lis aussi le papier en général
- Ensuite deep dive dans 1 partie du GPU opti
	- Tristan : 3.1 
	- Fotis : 3.2 et 3.3 
- Ds 1-2 semaines mise en commun et début des experiences gobales

### Meeting 30/11

Pour jeudi 
	- Finir partie Maths rapport
	- Finir opti background

### Meeting 09/12

- rajouter les MSE figures 1 et figures 3 
- changer les notations de M\alpha en H\alpha -> OK
- corriger l'équation de H^-1 -> OK
- expliciter plus FALKON CPU (our) vs FALKON GPU -> OK
- eclaircir le point de low density flop/memory kernel (lien avec "le same data is reused", il faut expliciter ce qui est réutilisé) -> OK
- enlever la subpartie 3.1 experiments -> OK
- rajouter une phrase en 3.1 sur le fait que même avec tous les efforts pour réduire la mémoire, il reste "150GO" sur les datasets, qui est juste énorme. -> OK
- rajouter le treatement de la diagonale -> OK
- citer la figure 6 et la commenter -> OK
- dans la figure 6 bien séparer les deux courbes | checker la valeur de m et la rajouter sur la figure | mettre moins de courbes, mettre deux~trois chunks max -> OK
- see Figure X ligne 115 -> OK
- corriger l'erreur de notaton cholesky -> OK
- insister dans la partie méthode sur les choses claires plutôt que celles des choses pas claires | peut être réexpliquer plus en détails le chunking | expliquer les figures, pas juste les mettres | passer un petit coup de ChatGPT de nouveau sur l'anglais | peut être enlever l'algo de Cholesky, en tout cas enlever la redondance avec les premières étapes -> OK

- bien vérifier la formule par batch block wise 
- rajouter l'opération par batch quand on a q,r et s
- terminer les heatmaps 
- expliquer le ratio ds 