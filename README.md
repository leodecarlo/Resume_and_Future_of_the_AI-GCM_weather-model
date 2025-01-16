# Resume_and_Future_of_the_AI-GCM_weather-model

The repository ports in Torch the model chosen from the ended project ["AI-GCM | The AI General Circulation Model"](https://www.cmcc.it/projects/ai-gcm-the-ai-general-circulation-model) and explain possible optimizations and evolutions of the architecture.

In the subfolder [Pytorch_porting_of_UNet-Illumia](Pytorch_porting_of_UNet-Illumia) there is the model plus the training pipeline. The training file implements Data Parallelism with a dynamic DataLoader and it is written with different  HPC implementations/communications.

In the subfolder [Architecture_Optimizations_and_Evolutions](Architecture_Optimizations_and_Evolutions) there is a [README.md](Architecture_Optimizations_and_Evolutions/README.md) describing the optimizations that can be done to the current architecture and how it could evolve to solve the problem of laziness in the weather forecasting.


