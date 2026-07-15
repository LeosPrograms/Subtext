# Subtext

A linguistic steganography tool built for the end-user. It is designed to encrypt messages into intelligible text, that could passably be produced by a human. Unlike traditional linguistic steganographic methods, which encode messages into existing text by word substitution, it uses a local large language model to generate text that contains the encoded and compressed stego (hidden message).


## Setup

**Requirements:** Python 3.9+

1. **Clone the repository**
   ```bash
   git clone https://github.com/LeosPrograms/Subtext
   cd Subtext
   ```

2. **Install dependencies**
   ```bash
   pip install torch transformers numpy
   ```

3. **Download the model weights**
   ```bash
   python download_model.py
   ```
   This downloads GPT-2 into `model/gpt2/` for offline use.

4. **Run the app**
   ```bash
   python app.py
   ```


## Acknowledgements

Subtext is an implementation of the arithmetic coding steganography algorithm given in Zachary Ziegler, Yuntian Deng, and Alexander Rush's ["Neural Linguistic Steganography" (2019)](https://arxiv.org/abs/1909.01496?utm_source=chatgpt.com).


## Similar Projects

- **[Neural Steganography](https://github.com/harvardnlp/NeuralSteganography)**

   A project that implements all three steganography methods from Zachary Ziegler, Yuntian Deng, and Alexander Rush's 2019 paper. 

   Differences: It is not open-source, and does not provide a graphical user interface.

- **[Textcoder](https://github.com/shawnz/textcoder)**

   An implementation of a distinct steganography method using LLM's, based on Matt Timmerman's [Bijective Arithmetic Coding algorithm](https://web.archive.org/web/20210901195459/http://www3.sympatico.ca/mt0000/biacode/).

   Differences: It supports authentication and encryption of the inputs, but provides inferior data efficiency. It lacks a graphical user interface.

- **[Tomato](https://github.com/user1342/Tomato)**

   A llm-based steganography tool that uses minimum-entropy coupling.

   Differences: It requires Nvidia CUDA, and is designed for high-end machines. It does not provide a graphical user interface.

- **[lm-steganography](https://github.com/falcondai/lm-steganography)**
   
   An implementation of Falcon Z. Dai and Zheng Cai's [Towards Near-imperceptible Steganographic Text](https://arxiv.org/abs/1907.06679) (2019).

   Differences: It is less efficient, and provides an inferior theoretical security guarantee, but may provide a more passable output. It does not provide a graphical user interface.
