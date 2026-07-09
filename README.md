# RQStyle

## Project Profile

This repository is used for presentations on **RQStyle: ResNet and Attention-Query Modulation for Structure-Preserving Style Transfer**. At this stage, it contains research code, experimental scripts, and partial evaluation utilities used during development. We will clean and reorganize the repository before the official release.

# RQStyle

Structure-preserving style transfer remains a challenging task in diffusion-based image generation. Existing stylization methods often achieve strong style appearance but may distort the content layout, object geometry, and fine semantic structures. This issue becomes more severe when the target style contains strong texture patterns or abstract visual rules, such as artistic paintings and Chinese paper-cut styles.

To address these challenges, we propose **RQStyle**, a structure-preserving style transfer framework based on **ResNet feature modulation** and **attention-query modulation**. Instead of relying only on global style injection, RQStyle explores how intermediate UNet residual features and attention query representations affect the balance between content structure and style appearance. By modulating these components, the method aims to preserve content layout and local structures while still transferring the reference style.

The project includes experiments on multiple style domains, including WikiArt-style transfer and Chinese paper-cut stylization. We also provide evaluation scripts for common metrics such as LPIPS, FID, and ArtFID, together with baseline comparisons against existing style transfer methods.

# Code

We will upload the cleaned training code, testing code, pretrained checkpoints, evaluation protocols, and full usage instructions at an appropriate time. The current repository is a working research version and may contain temporary experiment files.
