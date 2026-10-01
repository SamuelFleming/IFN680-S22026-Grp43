IFN680- Project 7 - Improving DDPM
Denoising diffusion probabilistic models (DDPM) 
In this task, you will explore the impact of noise scheduling on diffusion models.
Hypothesis to verify:
A cosine schedule allows a denoising diffusion probabilistic model to generate high-quality samples using fewer time steps than a linear schedule.
You will train two separate models to test this hypothesis. You will submit your training code, a 2-page report, and a reproducibility package containing an inference notebook that loads your pre-trained models to generate examples for all 10 digits.
Task details
Work through the following tasks:
Dataset setup
Download the following dataset: 
mnist_custom.pt
Download mnist_custom.pt


Custom MNIST dataset
For this challenge, you will not use the standard torchvision MNIST dataset. Instead, you are provided with a custom dataset file named mnist_custom.pt. You can load the dataset using PyTorch as follows: 
import torch 
data = torch.load('mnist_custom.pt') 
The dataset dictionary contains four keys that provide the necessary PyTorch tensors for the training images and their corresponding digit labels, as well as the separate testing images and labels. 
Task 1: Model training
Train 2 separate diffusion models in the DDPM_CosineSchedule.ipynb notebook on the mnist_custom.pt dataset: 
One model using the linear_beta_schedule that we defined in the 9.3 Tutorial. 
One model using your own implementation of a cosine_beta_schedule. 
Task 2: Comparison of the models (inference)
Create a separate inference: main_report.ipynb notebook that loads your two models from their saved checkpoints. Run the inference for both the Linear Schedule and the Cosine Schedule to visually compare the generation quality. Perform conditional sampling to generate images for all 10 digits (Classes 0–9). You must generate 12 samples per digit (a total of 120 images per model) to demonstrate consistency and variety.

Report
Submit a professional report in PDF format on this page. Your report should be clear, concise, and technically precise. It must include the following sections: 
Project 7: report Sections

Introduction:	
Explain the hypothesis and why adding a discriminator term might affect reconstruction quality. Include bibliographic references that attempted to prove this hypothesis and what they found. 
Methodology
Summarise the baseline training. Describe the modifications made to incorporate the cosine schedule and any corresponding changes to the training process.  
Experiments:
Describe what experiments you did to investigate that hypothesis. 
Discussion
Describe the results of your experiments, and analyse whether the results support the hypothesis. Discuss the impact of the cosine schedule on both training time and image quality. 
Conclusion:
Summarise your main findings in 2–3 sentences. Optionally, suggest further experiments or improvements. 
