<div algin="center">
# CoDy: Modeling Scholarly Collaboration and Temporal Dynamics in
Citation Networks for Impact Prediction

<a href='https://dl.acm.org/doi/10.1145/3626772.3657926'>
</div>

Environment
---
For pip: ``
<br>
`pip install requirements.txt`


Data Processing
---
The core process is to build the labels from academic graphs, including the future citations (like, in 5 years) and the colabration among authors. Refer to the `data_process` part for the details.

Framework
---
<img src="assets/framework.svg" alt="cody framework" width="800"/> 

Peformances
---
Performances on Impact Prediction Task. The best and second-best performances are indicated in bold and italicized, respectively. The bottom line indicates the percentage improvement of our model compared to the second-best one. ↑: larger value, better performance. ↓: smaller value, better performance.

| **Task**               | **Category**       | **Model**  | **APS MALE ↓** | **APS Log R2 ↑** | **DBLP MALE ↓** | **DBLP Log R2 ↑** | **APS Acc ↑** | **APS Macro-F1 ↑** | **DBLP Acc ↑** | **DBLP Macro-F1 ↑** |
|------------------------|--------------------|------------|----------------|------------------|-----------------|-------------------|---------------|---------------------|----------------|----------------------|
| **Citation Count Prediction** | Citation Model  | SciBERT    | 0.5220         | 0.4231           | 0.8267          | 0.2610            | 64.71         | 61.78               | 69.74          | 64.45               |
|                        |                    | HINTS      | 1.0230         | -0.1481          | 0.7645          | 0.2990            | 57.23         | 53.11               | 63.81          | 43.55               |
| **Citation Level Classification** | Cascade Model  | GTGCN      | 1.2405         | -0.2128          | 0.9292          | -0.0793           | 59.21         | 47.78               | 67.86          | 34.44               |
|                        |                    | CCGL       | 0.8731         | 0.3145           | 0.7812          | 0.2778            | 54.63         | 47.24               | 73.58          | 51.13               |
|                        |                    | MUCas      | 0.6321         | 0.4631           | 0.5110          | 0.6799            | 63.77         | 61.83               | 83.09          | 72.08               |
|                        | Dynamic GNN        | HGT        | 0.5411         | 0.4635           | 0.5031          | 0.6873            | 61.64         | 60.86               | 80.91          | 72.08               |
|                        |                    | EGCN       | 0.6897         | 0.3238           | 0.8758          | 0.1182            | 56.37         | 54.76               | 75.64          | 39.72               |
|                        |                    | ROLAND     | 1.1732         | -0.1502          | 1.2492          | -0.5857           | 53.24         | 50.10               | 46.57          | 36.30               |
|                        |                    | H2CGL      | 0.3941         | 0.5150           | 0.4540          | 0.7548            | *65.12*       | *63.37*             | *84.52*        | 80.50               |
|                        |                    | STHN       | *0.3823*       | *0.5289*         | *0.4510*        | *0.7587*          | 63.87         | 61.63               | 82.96          | *80.73*             |
|                        | Ours               | CoDy       | **0.3587**     | **0.5457**       | **0.4237**      | **0.7847**        | **66.23**     | **66.56**           | **86.15**      | **84.81**           |
|                        |                    | % Improve  | 6.17% ↓        | 3.18% ↑          | 6.05% ↓         | 3.43% ↑           | 1.70% ↑       | 5.03% ↑             | 3.68% ↑        | 5.05% ↑             |
