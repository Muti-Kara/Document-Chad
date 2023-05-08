# Document-Chad: Gradio Document Question-Answering Project

## Overview

This project aims to create a user-friendly interface with Gradio that allows users to upload a document and then ask questions about the content of the document. By leveraging the Pinecone and OpenAI APIs, the system will attempt to provide accurate and relevant answers to the questions. Please note that this project is currently **incomplete** and under development. We appreciate your patience and understanding.

## Getting Started

### Prerequisites

Before you can run the project, you need to have the following dependencies installed:

- Python 3.6 or later
- Gradio
- Pinecone
- OpenAI
- NLTK
- Tiktoken

You can install the required packages using `pip`:

```bash
pip install -r requirements.txt
```

### API Keys

To use the Pinecone and OpenAI APIs, you need to obtain the respective API keys. Please register or log in to their websites to get the keys:

- [Pinecone API key](https://www.pinecone.io/)
- [OpenAI API key](https://beta.openai.com/signup/)

After obtaining the keys, create a `.env` file in the project directory and add your keys as follows:

```
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=your_pinecone_env
OPENAI_API_KEY=your_openai_api_key
```

### Installation

1. Clone the repository to your local machine:

```bash
git clone https://github.com/Muti-Kara/Document-Chad.git
```

2. Navigate to the project directory:

```bash
cd Document-Chad
```

3. Run the application:

```bash
python app.py
```

### Usage

1. After launching the application, open your web browser and navigate to `http://127.0.0.1:7860/`. You should see the Gradio interface.

2. Upload the document you want to analyze by clicking the "Upload" button and selecting the file from your computer. The system currently supports only .pdf format.

3. Enter your question in the provided text box.

4. Click "Submit" to obtain the answer. The system will analyze the document and attempt to provide an accurate answer based on the content.

5. If you want to ask another question, simply enter the new question and click "Submit" again.

## Contributing

Please note that this project is still under development, and contributions are more than welcome. If you would like to contribute, please follow these steps:

1. Fork the repository on GitHub.
2. Create a new branch for your feature or bugfix.
3. Commit your changes, ensuring that you have added or modified the necessary tests and documentation.
4. Submit a pull request to the main branch.

## License

This project is licensed under the GPLv3 License. See the [LICENSE](LICENSE) file for more information.

## Acknowledgements

- Gradio team for creating a great platform for creating user-friendly interfaces.
- Pinecone and OpenAI for providing powerful APIs used in this project.
- All the contributors who helped in shaping this project.

## Contact

If you have any questions or suggestions, please feel free to open an issue or reach out to the project maintainer at [youremail@example.com](mailto:youremail@example.com).
