# Project 7 — Final Training Run Notes

Before producing the final 150-epoch outputs:

- Set `EPOCHS = 150`.
- Use **Restart Kernel → Run All** so both models are trained from a clean notebook state.
- Confirm both models still begin from the intended identical initialisation.
- Consider resetting the random seeds immediately before each model's training call so DataLoader shuffling, sampled timesteps, and Gaussian-noise draws are as controlled as practical between the Linear and Cosine experiments.
- Strict bit-for-bit CUDA determinism is **not required** unless reproducibility becomes an issue; seeded experimental control is sufficient for this assessment.
- Confirm both final checkpoints report `epochs = 150` and overwrite the temporary 50-epoch checkpoints.
- Rerun all diagnostics/inference used in the report after final training; do not report conclusions from the temporary 50-epoch development models.



Here is what ChatGPT said initially
> Copied straight out of chat, so alot of the markdown formatting has been lost

```
One thing I think we should improve
Our current setup guarantees identical initial weights, but it doesn't completely isolate the training randomness of Model A and Model B.
Remember that training itself does this:
t = torch.randint(...)


and q_sample() does:
noise = torch.randn_like(x_start)


while the DataLoader is shuffled.
So there are three stochastic pieces:
batch ordering
random timestep t
random Gaussian ε

Ideally, the linear and cosine experiments should receive equivalent random sequences so the principal changed variable is:
linear schedule  vs  cosine schedule

We can accomplish that quite simply.
Immediately before each final training call, reset the RNGs:
def reset_random_seeds(seed=SEED):    random.seed(seed)    np.random.seed(seed)    torch.manual_seed(seed)    if torch.cuda.is_available():        torch.cuda.manual_seed(seed)        torch.cuda.manual_seed_all(seed)


Then Model A:
reset_random_seeds()linear_history = train_ddpm(    model=linear_model,    train_loader=train_loader,    schedule=linear_schedule,    epochs=EPOCHS,    learning_rate=LEARNING_RATE,    device=device)


And Model B:
reset_random_seeds()cosine_history = train_ddpm(    model=cosine_model,    train_loader=train_loader,    schedule=cosine_schedule,    epochs=EPOCHS,    learning_rate=LEARNING_RATE,    device=device)


That's a worthwhile improvement before your final 150-epoch run.
```