# SmolLM2 Text Generator

This is a Gradio application for generating text using the trained SmolLM2 model. The app allows users to input a text prompt and generate multiple sequences of text based on that prompt. The number of sequences and the length of the generated text can be adjusted using sliders.

## Features

- **Text Generation**: Generate text based on a user-provided prompt using the SmolLM2 model.
- **Adjustable Length**: Control the length of the generated text.
- **Multiple Sequences**: Generate multiple sequences of text in one go.

## Requirements

To run this application, you need the following Python packages:

- `torch`
- `transformers`
- `gradio`

You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Usage

1. **Run the App**: Launch the Gradio app by running the following command in your terminal:

   ```bash
   python app.py
   ```

2. **Input Prompt**: Enter your desired text prompt in the provided textbox.

3. **Adjust Sliders**:
   - Use the "Predict Additional Text of Length" slider to set the desired length of the generated text.
   - Use the "Number of Sequences to Generate" slider to specify how many sequences you want to generate.

4. **Generate Text**: Click the "Generate Text" button to produce the text sequences.

5. **View Output**: The generated sequences will be displayed in the output textbox, each prefixed with "Sequence X:" for clarity.

## Example

- **Prompt**: "Once upon a time"
- **Number of Sequences**: 2

**Output**:
```
Sequence 1:
Once upon a time, there is a cat ....

Sequence 2:
Once upon a time in a small village ....
```

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments

- Hugging Face for the Transformers library and model support.
- Gradio for providing an easy-to-use interface for machine learning applications.
- The SmolLM2 model for enabling advanced text generation capabilities. 