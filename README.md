# LLaMA 3.2 Arabic-English Math QA Fine-tuning

Fine-tuning Meta's LLaMA 3.2 3B Instruct model for bilingual (Arabic/English) mathematical question answering using QLoRA (4-bit quantization + LoRA).

## Project Overview

This project fine-tunes LLaMA 3.2 3B Instruct on a bilingual Arabic-English mathematical question-answering dataset. The model learns to understand and respond to math questions in both languages using efficient parameter-efficient fine-tuning (PEFT) techniques.

### Key Features
- **Bilingual Support**: Handles both Arabic and English mathematical queries
- **Memory Efficient**: QLoRA with 4-bit quantization
- **Fast Training**: LoRA adapters train 95% faster than full fine-tuning

## Dataset

The model is trained on a curated dataset of mathematical questions covering:
- Arithmetic operations
- Algebra
- Geometry
- Word problems
- Mixed bilingual contexts
