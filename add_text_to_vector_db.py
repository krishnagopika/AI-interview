from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import uuid
from dotenv import load_dotenv

load_dotenv()

# Initialize Pinecone
pc = Pinecone()

# Create index (if not already created)
index_name = "tech-questions"
if index_name not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,  # embedding size for MiniLM-L6-v2
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to the index
index = pc.Index(index_name)


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

import uuid

data = [
    # ===================== JAVA =====================
    {
        "id": str(uuid.uuid4()),
        "question": "What is the difference between JDK, JRE, and JVM?",
        "answer": "JDK is a development kit, JRE is the runtime environment, and JVM executes bytecode.",
        "difficulty": "Easy",
        "skill": "Java Full Stack",
        "unit": "Java"
    },
    {
        "id": str(uuid.uuid4()),
        "question": "Explain how HashMap works internally in Java.",
        "answer": "HashMap uses an array of buckets and linked lists or trees for key-value pairs, using hashcodes.",
        "difficulty": "Medium",
        "skill": "Java Full Stack",
        "unit": "Java"
    },
    {
        "id": str(uuid.uuid4()),
        "question": "How does the JVM manage memory through the heap, stack, and garbage collection?",
        "answer": "The JVM allocates objects on the heap, uses the stack for method frames, and cleans unused objects via garbage collection.",
        "difficulty": "Hard",
        "skill": "Java Full Stack",
        "unit": "Java"
    },

    # ===================== SPRING BOOT =====================
    {
        "id": str(uuid.uuid4()),
        "question": "How does Spring Boot simplify Java development?",
        "answer": "It provides auto-configuration, embedded servers, and starter dependencies for rapid development.",
        "difficulty": "Easy",
        "skill": "Java Full Stack",
        "unit": "Spring Boot"
    },
    {
        "id": str(uuid.uuid4()),
        "question": "What is dependency injection in Spring Boot and why is it used?",
        "answer": "Dependency injection allows Spring to manage object creation and dependencies, promoting loose coupling.",
        "difficulty": "Medium",
        "skill": "Java Full Stack",
        "unit": "Spring Boot"
    },
    {
        "id": str(uuid.uuid4()),
        "question": "How does Spring Boot handle microservices communication and discovery?",
        "answer": "Through Spring Cloud components like Eureka for discovery and Feign or RestTemplate for communication.",
        "difficulty": "Hard",
        "skill": "Java Full Stack",
        "unit": "Spring Boot"
    },

    # ===================== REACT =====================
    {
        "id": str(uuid.uuid4()),
        "question": "What is JSX in React?",
        "answer": "JSX is a syntax extension that allows writing HTML-like code inside JavaScript.",
        "difficulty": "Easy",
        "skill": "Frontend Development",
        "unit": "React"
    },
    {
        "id": str(uuid.uuid4()),
        "question": "Explain the difference between functional and class components in React.",
        "answer": "Functional components are stateless and use hooks, while class components have lifecycle methods and state by default.",
        "difficulty": "Medium",
        "skill": "Frontend Development",
        "unit": "React"
    },
    {
        "id": str(uuid.uuid4()),
        "question": "How does React's reconciliation algorithm optimize rendering?",
        "answer": "It uses the virtual DOM to compare and efficiently update only changed parts of the UI.",
        "difficulty": "Hard",
        "skill": "Frontend Development",
        "unit": "React"
    },

    # ===================== CI/CD =====================
    {
        "id": str(uuid.uuid4()),
        "question": "What does CI/CD stand for?",
        "answer": "CI stands for Continuous Integration and CD stands for Continuous Deployment or Delivery.",
        "difficulty": "Easy",
        "skill": "DevOps",
        "unit": "CI/CD"
    },
    {
        "id": str(uuid.uuid4()),
        "question": "Explain the difference between Continuous Deployment and Continuous Delivery.",
        "answer": "Continuous Delivery ensures code is always deployable, while Continuous Deployment automatically deploys every change to production.",
        "difficulty": "Medium",
        "skill": "DevOps",
        "unit": "CI/CD"
    },
    {
        "id": str(uuid.uuid4()),
        "question": "How do you implement blue-green deployment in CI/CD pipelines?",
        "answer": "By running two environments (blue and green), switching traffic to the new version after verification to minimize downtime.",
        "difficulty": "Hard",
        "skill": "DevOps",
        "unit": "CI/CD"
    },

    # ===================== DEVOPS =====================
    {
        "id": str(uuid.uuid4()),
        "question": "What is infrastructure as code (IaC)?",
        "answer": "IaC means managing and provisioning infrastructure using code instead of manual processes.",
        "difficulty": "Easy",
        "skill": "DevOps",
        "unit": "DevOps"
    },
    {
        "id": str(uuid.uuid4()),
        "question": "Explain the difference between Docker and Kubernetes.",
        "answer": "Docker is for containerization, while Kubernetes is for orchestrating and managing containers at scale.",
        "difficulty": "Medium",
        "skill": "DevOps",
        "unit": "DevOps"
    },
    {
        "id": str(uuid.uuid4()),
        "question": "How does monitoring and alerting integrate into DevOps pipelines?",
        "answer": "Using tools like Prometheus, Grafana, and ELK stack to track metrics, detect issues, and trigger automated alerts.",
        "difficulty": "Hard",
        "skill": "DevOps",
        "unit": "DevOps"
    },

    # ===================== LLMs =====================
    {
        "id": str(uuid.uuid4()),
        "question": "What is a Large Language Model (LLM)?",
        "answer": "An LLM is a transformer-based model trained on massive text data to understand and generate human-like language.",
        "difficulty": "Easy",
        "skill": "AI/ML",
        "unit": "LLMs"
    },
    {
        "id": str(uuid.uuid4()),
        "question": "What is the role of tokenization in LLMs?",
        "answer": "Tokenization splits text into smaller units (tokens) that the model can process and learn from.",
        "difficulty": "Medium",
        "skill": "AI/ML",
        "unit": "LLMs"
    },
    {
        "id": str(uuid.uuid4()),
        "question": "Explain the concept of attention in transformer-based LLMs.",
        "answer": "Attention mechanisms allow the model to weigh different words based on their relevance to each other during training.",
        "difficulty": "Hard",
        "skill": "AI/ML",
        "unit": "LLMs"
    },

    # ===================== PROMPT ENGINEERING =====================
    {
        "id": str(uuid.uuid4()),
        "question": "What is prompt engineering?",
        "answer": "Prompt engineering is designing input prompts to guide large language models to produce desired outputs.",
        "difficulty": "Easy",
        "skill": "AI/ML",
        "unit": "Prompt Engineering"
    },
    {
        "id": str(uuid.uuid4()),
        "question": "What are few-shot and zero-shot prompting techniques?",
        "answer": "Few-shot uses examples in the prompt for context; zero-shot relies solely on the prompt without examples.",
        "difficulty": "Medium",
        "skill": "AI/ML",
        "unit": "Prompt Engineering"
    },
    {
        "id": str(uuid.uuid4()),
        "question": "How can prompt injection attacks affect LLM-based applications?",
        "answer": "They manipulate prompts to override instructions or extract sensitive information, leading to security risks.",
        "difficulty": "Hard",
        "skill": "AI/ML",
        "unit": "Prompt Engineering"
    }
]

# Embed all questions
texts = [d["question"] for d in data]
embeddings = model.encode(texts).tolist()

# Upsert into Pinecone
vectors = []
for i, d in enumerate(data):
    vectors.append({
        "id": d["id"],
        "values": embeddings[i],
        "metadata": {
            "question": d["question"],
            "answer": d["answer"],
            "difficulty": d["difficulty"],
            "skill": d["skill"],
            "unit": d["unit"]
        }
    })

index.upsert(vectors=vectors)
print("✅ Data inserted into Pinecone successfully.")
