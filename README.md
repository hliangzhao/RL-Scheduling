## REINFORCE and GNN-based Scheduler

I try to re-implement the algorithm **Decima**, published in SIGCOMM '19 
(https://web.mit.edu/decima/). The official code its authors provide is written in tensorflow v1 
and is a little bit confusing :-(. I try to re-implement it with torch.

Also, **Decima** is applied for stream processing framework, such as Spark (https://github.com/apache/spark) 
and Flink (https://github.com/apache/flink). I try to adapt this algorithm to batch systems, such as 
Flink, etc.

By the way, the authors do not open the codes for multi-resource scheduling. I may implement it by myself in this repos.

It's also interesting to adapt **Decima** to K8S-based scheduling frameworks, such as Volcano (https://github.com/volcano-sh/volcano). 
However, we have to face many new challenges. For example, what can be scheduled is not the jobs themselves, 
but the pods which wrap them. However, we don't know how the jobs maps to the pods. 

I'm working on it now.