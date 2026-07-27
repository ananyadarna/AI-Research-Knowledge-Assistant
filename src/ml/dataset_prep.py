import os
import csv
import random

CATEGORIES = [
    "Artificial Intelligence",
    "Machine Learning",
    "Computer Vision",
    "Natural Language Processing",
    "Robotics",
    "Cyber Security",
    "Cloud Computing"
]

KEYWORDS = {
    "Artificial Intelligence": [
        "artificial intelligence", "general intelligence", "agi", "expert systems", "heuristics",
        "intelligent agent", "knowledge representation", "semantic web", "symbolic logic", "search algorithms",
        "turing test", "cognitive simulation", "problem solving system", "reasoning engine", "knowledge base"
    ],
    "Machine Learning": [
        "machine learning", "supervised learning", "unsupervised learning", "reinforcement learning",
        "random forest", "gradient boosting", "neural networks", "linear regression", "support vector machines",
        "overfitting", "training set", "cross validation", "loss function", "k-means clustering", "decision trees"
    ],
    "Computer Vision": [
        "computer vision", "image processing", "object detection", "image segmentation", "convolutional neural network",
        "cnn", "optical flow", "facial recognition", "yolo", "edge detection", "feature extraction", "opencv",
        "pixel intensities", "depth estimation", "camera calibration"
    ],
    "Natural Language Processing": [
        "natural language processing", "nlp", "text tokenization", "part of speech tagging", "named entity recognition",
        "sentiment analysis", "word embeddings", "transformer model", "bert", "gpt", "sequence to sequence",
        "language modeling", "machine translation", "syntactic parsing", "stopword removal"
    ],
    "Robotics": [
        "robotics", "autonomous navigation", "robotic kinematics", "actuators and sensors", "manipulator arm",
        "lidar mapping", "slam algorithm", "obstacle avoidance", "motion planning", "humanoid robot",
        "feedback loop control", "odometry", "ros node", "teleoperation", "quadcopter flight"
    ],
    "Cyber Security": [
        "cyber security", "cryptographic protocols", "malware signature", "intrusion detection system", "firewall policy",
        "sql injection vulnerability", "phishing attack prevention", "ransomware decryption", "zero day exploit",
        "multifactor authentication", "penetration testing", "buffer overflow", "symmetric encryption", "ddos mitigation",
        "security audit log"
    ],
    "Cloud Computing": [
        "cloud computing", "amazon web services aws", "microsoft azure", "virtualization hypervisor", "serverless lambda",
        "docker container", "kubernetes orchestration", "microservices architecture", "scalability and elastic load balancing",
        "software as a service saas", "infrastructure as a service iaas", "cloud database", "tenant isolation", "devops pipeline"
    ]
}

TEMPLATES = [
    "This research paper discusses the application of {} in solving complex real-world challenges.",
    "A comprehensive study on {}, focusing on state-of-the-art architectures and validation datasets.",
    "We present a novel approach utilizing {} to optimize performance metrics and accuracy bounds.",
    "The implementation of {} has shown significant improvements over traditional base configurations.",
    "Evaluating the robustness of {} under varying operational constraints and high throughput scenarios.",
    "Recent advancements in {} have led to widespread adoption in industry and academia.",
    "Our model integrates {} to improve reliability and decrease latency in production pipelines.",
    "In this paper, we explore how {} can be scaled to support large-scale distributed architectures.",
    "A review of standard practices in {} reveals several open problems regarding security and validation.",
    "Developing system prototypes based on {} is crucial for next-generation automated systems."
]

def generate_synthetic_text(category: str, num_samples: int = 150) -> list[str]:
    samples = []
    keywords = KEYWORDS[category]
    for _ in range(num_samples):
        # Pick 3 or 4 random keywords
        chosen_keywords = random.sample(keywords, k=min(4, len(keywords)))
        # Fill templates
        sentences = []
        for kw in chosen_keywords:
            temp = random.choice(TEMPLATES)
            sentences.append(temp.format(kw))
        # Combine sentences into a paragraph
        random.shuffle(sentences)
        samples.append(" ".join(sentences))
    return samples

def prepare_dataset(output_path: str = "./data/dataset/training_data.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        
        for idx, category in enumerate(CATEGORIES):
            texts = generate_synthetic_text(category, num_samples=200)
            for text in texts:
                writer.writerow([text, idx])
                
    print(f"Dataset preparation complete. Saved to {output_path}")

if __name__ == "__main__":
    prepare_dataset()
