<h1 align="center"> 🔍 Hybrid Deep Searcher:</br>Integrating Parallel and Sequential Search Reasoning</a></h1>

<div align="center"> 
![Static Badge](https://img.shields.io/badge/HuggingFace-Dataset-red)
[![Static Badge](https://img.shields.io/badge/HuggingFace-Model-blue)
](https://huggingface.co/dayoon/HybridDeepSearcher) 
<!-- [![Paper](https://img.shields.io/badge/Paper-arXiv-b5212f.svg?logo=arxiv)](https://arxiv.org/abs/2501.05366)  -->
</div>

<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

## 💡 Overview

Previous single-query iterative methods suffer from high latency, inefficient workflows, and poor scalability in exhaustive searching across a large number of documents. In contrast, this work empowers an LRM to integrate parallel and sequential search by training it to distinguish between parallelizable and sequential queries. This integration of parallel and sequential search improves both efficiency and accuracy.

### ✨ Dataset

Existing multi-step QA datasets focus mainly on sequential retrieval, leaving parallelizable scenarios underexplored. To fill this gap, this work introduces HDS-QA, a dataset of hybrid-hop questions that mix parallelizable and sequential subqueries. It also includes synthetic answer trajectories that guide models through reasoning, querying, and retrieval loops involving parallel queries.

### ✨ Method

The authors present HybridDeepSearcher, an LRM fine-tuned on HDS-QA that integrates parallel querying into sequential reasoning, reducing iterations and improving coherence through visualized reasoning and planning steps. Experiments on a subset of BrowseComp show it achieves higher F1 scores with fewer sequential searches and API calls than baselines, while scaling effectively as the search budget increases. Its dynamic retrieval strategies and adaptive workflows enable efficient handling of large document sets for complex questions.

---

## 🏃 Quick Start

### 🔧 Environment Setup
```bash
cd HybridDeepSearcher
conda env create -f environment.yml
conda activate hybridds
pip install ms-swift==3.5.3
```

### ✨ Data Generation

### ✨ Train
```bash
cd train
bash train.sh
```

### ✨ Model Inference

```bash
cd infer
bash inference.sh # Run inference
python eval.py -datasets musique fanoutqa frames med_browse_comp browse_comp # Evaluation
```

<!-- 
## 📄 Citation

If you find this work helpful, please cite our paper:
```bibtex
``` -->