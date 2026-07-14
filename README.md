# Subtext

## Setup

**Requirements:** Python 3.9+

1. **Clone the repository**
   ```bash
   git clone <repo-url>
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