IFN680- Project 6 - The sharpness quest
The sharpness quest
In this assessment, you will investigate how modifying the training objective of a Conditional Variational Autoencoder (cVAE) affects the quality of its reconstructions. Starting from the implementation in Tutorial 8.3, you will extend the baseline model and experimentally evaluate the following hypothesis: 
Adding a discriminator term to the loss function of a cVAE during training increases the sharpness of the reconstructed images.
To test this, you will incorporate the additional loss component into the training process and compare the reconstructions produced by the standard and modified models. You will then analyse the results to determine whether the observed behaviour supports the hypothesis and discuss the impact of this change on reconstruction quality. 
Task details
To investigate the hypothesis, you will extend the cVAE implementation developed during Tutorial 8.3 and perform a comparative experiment. 
Work through the following tasks to complete this challenge:
Data
Download the following dataset for this challenge: 
mnist_custom.pt
Download mnist_custom.pt
.
Task 1: Train a baseline VAE
Use the modified MNIST dataset provided and train the standard cVAE presented in the tutorial. Ensure the model is trained until convergence and can generate reasonable samples. It should use 7 latent dimensions. This model will serve as the baseline for comparison. 
Task 2: Extend the VAE with a discriminator loss
Modify the training objective by introducing a discriminator-based loss term. This may require implementing a discriminator network and adapting the training procedure so that the additional loss contributes to the optimisation of the cVAE. Train the modified model to convergence using the same dataset and comparable training settings. 


Task 3: Evaluate and compare models
Run inference using both the standard cVAE and the modified cVAE with discriminator loss. Perform conditional sampling for all 10 digit classes (Classes 0–9), generating 12 samples per digit, for a total of 120 generated images per model. Use these samples to visually compare the quality, sharpness, consistency, and variety of the generated digits.

Next, quantitatively compare the two cVAE variants to qunatitatively assess whether the discriminator loss improves the quality or sharpness of the generated images. Possible evaluation approaches include Fréchet Inception Distance (FID), image sharpness metrics such as Laplacian variance, or a standalone digit classifier to measure how readily the generated digits can be recognised.
Finally, analyse the visual and quantitative results and discuss whether the evidence supports the hypothesis that incorporating the discriminator loss improves the sharpness or overall quality of the generated images.


Submission Instructions
Report
Submit a professional report in PDF format (max 2 pages) on this page. Your report should be clear, concise, and technically precise. It must include the following sections: 
Project Code
You will submit a Zipped folder named project6_code.zip on this page. This folder must include:
cVAE_DiscriminatorLoss.ipynb: A Jupyter Notebook containing the full implementation of the proposed hypothesis model: a cVAE trained with an additional discriminator loss.  
main_report.ipynb: You must submit a ZIP file containing an additional notebook (.ipynb) that reproduces all the plots, numerical, and qualitative results presented in your report so the grader can verify reproducibility. Do not forget to include the 12×10 grid of generated digits (12 samples per digit, 120 images) for both models. Also, include any auxiliary files required for this process, such as pickle files with losses, model weights, or other necessary data. Ensure this notebook with these auxiliary files run in IFN680 computing environment. This notebook will be run for grading, failure on any of the cell will results in 0 mark for the code evaluation.  


Project 6: report Sections

Introduction:	
Explain the hypothesis and why adding a discriminator term might affect reconstruction quality. Include bibliographic references that attempted to prove this hypothesis and what they found.
Methodology
Summarise the baseline cVAE training. Describe the modifications made to incorporate the discriminator loss and any corresponding changes to the training process.
Experiments:
Describe the experiments you performed to compare the baseline and modified cVAE. Include the evaluation metrics used (e.g., inception distance, sharpness metrics, or digit classifier).
Discussion
Present the key results, including generated images. Include quantitative and qualitative evaluations. Analyse whether the results support the hypothesis. Discuss the impact of the discriminator loss on reconstruction sharpness and overall image quality. Reflect on any limitations or factors influencing the results.
Conclusion:
Summarise your main findings in 2–3 sentences. Optionally, suggest further experiments or improvements.


Supporting resources
PyTorch documentation
Links to an external site.
 (PyTorch, n.d.).
Use the IFN680 GPU environment. Ensure your code is configured to utilise the available hardware via .to(device) to ensure your experiments run within a reasonable timeframe. 
