## REINFORCE and GNN-based Scheduler

In this repos I try to re-implement the algorithm **Decima**, published in SIGCOMM 2019 
(https://web.mit.edu/decima/). The official code its authors provide is written in tensorflow v1 
and is a little bit confusing :-(. I try to re-implement it with torch.

Also, **Decima** is applied for stream processing framework, such as Spark (https://github.com/apache/spark) 
and Flink (https://github.com/apache/flink). I try to adapt this algorithm to batch systems, such as 
Flink and Volcano (https://github.com/volcano-sh/volcano), etc.

By the way, the authors do not open the codes for multi-resource scheduling. I may implement it by myself in this repos.