# Robotics QA Bot

domain-specific LLM-powered question-answering assistant designed to answer technical queries about robotics systems, including:
- Robot Operating System (ROS)
- Sensors (LiDAR, IMU, etc.)
- SLAM (Simultaneous Localization and Mapping)
- Navigation and Perception
The assistant uses a collection of robotics-related PDFs for context and retrieves relevant answers using semantic search and language models.

# Documents
All robotics reference PDFs are stored in the docs/ folder. These are indexed and queried by the QA engine for context-aware answers.

# Technologies Used
- Python
- LangChain / Transformers
- FAISS or ChromaDB for vector search
- Streamlit / Flask (depending on app interface)
- PDF document loaders and embedders

# To Do
- Add support for multilingual queries.
- Dockerize for easy deployment.
- Add unit tests.
- Fine-tune the language model

# Author
Darshan Bakilana Ramesh
https://www.linkedin.com/in/darshanbakilanaramesh/
https://portfolio-darshan-bakilana-rameshs-projects.vercel.app/
