## REINFORCE and GNN-based Scheduler

I try to re-implement the algorithm **Decima**, published in SIGCOMM '19 
(https://web.mit.edu/decima/). The official code its authors provide is written in tensorflow v1 
and is a little confusing.

By the way, the authors do not open the codes for multi-resource scheduling (and the interface to Spark). 
I may implement it by myself in this repos.

It's also interesting to adapt **Decima** to K8S-based scheduling frameworks, such as Volcano (https://github.com/volcano-sh/volcano). 
However, we have to face many new challenges. For example, what can be scheduled is not the jobs themselves, 
but the pods which wrap them.  Cross-server communication overhead also need to be taken into consideration.
