"""
Knowledge Base — ChromaDB vector store with domain-tagged interview questions.
Seeds the database with questions across multiple technology domains.
Uses custom sentence-transformer embeddings for high-quality semantic search.
"""

import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

_collection = None
_client = None

# ---------- Comprehensive interview question knowledge base ----------
KNOWLEDGE_BASE = [
    # ======== Python ========
    {"question": "Explain the difference between a list and a tuple in Python. When would you use each?", "domain": "python", "difficulty": "easy", "answer_key": "Lists are mutable, tuples immutable. Use tuples for fixed data, lists for dynamic collections.", "tags": ["data-structures", "mutability"]},
    {"question": "What are Python decorators and how do they work internally?", "domain": "python", "difficulty": "medium", "answer_key": "Decorators are functions that modify other functions. They use closure and higher-order function concepts.", "tags": ["decorators", "closures", "higher-order"]},
    {"question": "Explain Python's GIL. How does it affect multi-threaded programs?", "domain": "python", "difficulty": "hard", "answer_key": "GIL prevents true parallel execution of Python bytecode. Use multiprocessing for CPU-bound tasks.", "tags": ["concurrency", "GIL", "threading"]},
    {"question": "What is the difference between deep copy and shallow copy in Python?", "domain": "python", "difficulty": "medium", "answer_key": "Shallow copy copies references, deep copy creates independent copies of nested objects.", "tags": ["copy", "references", "objects"]},
    {"question": "Explain generators in Python. How are they different from regular functions?", "domain": "python", "difficulty": "medium", "answer_key": "Generators use yield to produce values lazily, saving memory for large sequences.", "tags": ["generators", "yield", "lazy-evaluation"]},
    {"question": "What are metaclasses in Python and when would you use them?", "domain": "python", "difficulty": "hard", "answer_key": "Metaclasses define how classes behave. Use for ORM frameworks, API validation, singleton patterns.", "tags": ["metaclasses", "OOP", "advanced"]},
    {"question": "Explain the difference between __str__ and __repr__ in Python.", "domain": "python", "difficulty": "easy", "answer_key": "__str__ for user-friendly display, __repr__ for unambiguous developer representation.", "tags": ["dunder", "string-representation"]},
    {"question": "What is monkey patching in Python? Give practical use cases.", "domain": "python", "difficulty": "medium", "answer_key": "Modifying classes/modules at runtime. Used in testing (mocking), hot fixes, extending libraries.", "tags": ["monkey-patching", "runtime", "testing"]},
    {"question": "What are context managers in Python? How do you create one?", "domain": "python", "difficulty": "medium", "answer_key": "Context managers handle setup/teardown via __enter__/__exit__. Create with class or @contextmanager decorator.", "tags": ["context-manager", "with-statement", "resource-management"]},
    {"question": "Explain the asyncio event loop in Python. How does async/await work?", "domain": "python", "difficulty": "hard", "answer_key": "Event loop manages coroutines cooperatively. await yields control back to the loop for I/O-bound concurrency.", "tags": ["asyncio", "coroutines", "async"]},

    # ======== DSA ========
    {"question": "Explain the time complexity of quicksort in best, average, and worst cases.", "domain": "dsa", "difficulty": "medium", "answer_key": "Best/Average: O(n log n), Worst: O(n²) when pivot is always min/max.", "tags": ["sorting", "quicksort", "complexity"]},
    {"question": "What is a hash table? How do you handle collisions?", "domain": "dsa", "difficulty": "medium", "answer_key": "Array with hash function mapping. Collisions handled by chaining or open addressing.", "tags": ["hashing", "collision-resolution"]},
    {"question": "Explain the difference between BFS and DFS. When would you use each?", "domain": "dsa", "difficulty": "medium", "answer_key": "BFS uses queue (shortest path), DFS uses stack (topological sort, cycle detection).", "tags": ["graphs", "BFS", "DFS", "traversal"]},
    {"question": "What is dynamic programming? Explain with an example.", "domain": "dsa", "difficulty": "hard", "answer_key": "Solving problems by breaking into overlapping subproblems. Example: Fibonacci, knapsack.", "tags": ["dynamic-programming", "optimization"]},
    {"question": "Explain the difference between a stack and a queue. Give real-world applications.", "domain": "dsa", "difficulty": "easy", "answer_key": "Stack is LIFO (undo operations), Queue is FIFO (task scheduling).", "tags": ["stack", "queue", "linear-structures"]},
    {"question": "What is a balanced BST? Compare AVL and Red-Black trees.", "domain": "dsa", "difficulty": "hard", "answer_key": "AVL is strictly balanced (faster lookups), Red-Black is relaxed (faster insertions/deletions).", "tags": ["BST", "AVL", "red-black", "balanced-trees"]},
    {"question": "Explain the concept of amortized time complexity with an example.", "domain": "dsa", "difficulty": "hard", "answer_key": "Average cost per operation over sequence. Example: dynamic array append is O(1) amortized.", "tags": ["amortized", "complexity-analysis"]},
    {"question": "How does a trie data structure work? What are its advantages over hash tables for string problems?", "domain": "dsa", "difficulty": "medium", "answer_key": "Trie stores characters at each node. Supports prefix search in O(k). Better for autocomplete, spell-check.", "tags": ["trie", "prefix-tree", "strings"]},
    {"question": "Explain Dijkstra's algorithm and its limitations. When would you use Bellman-Ford instead?", "domain": "dsa", "difficulty": "hard", "answer_key": "Dijkstra uses greedy approach, O(V+E log V). Fails with negative edges. Bellman-Ford handles negative weights.", "tags": ["graphs", "shortest-path", "dijkstra", "bellman-ford"]},
    {"question": "What is a heap? Explain how heapify works and its time complexity.", "domain": "dsa", "difficulty": "medium", "answer_key": "Complete binary tree with heap property. Heapify is O(n) bottom-up. Insert/extract is O(log n).", "tags": ["heap", "priority-queue"]},

    # ======== System Design ========
    {"question": "How would you design a URL shortener like bit.ly?", "domain": "system_design", "difficulty": "medium", "answer_key": "Hash-based ID generation, distributed key-value store, redirect service, analytics.", "tags": ["url-shortener", "hashing", "scalability"]},
    {"question": "Explain the CAP theorem and its implications for distributed systems.", "domain": "system_design", "difficulty": "hard", "answer_key": "Cannot have all three: Consistency, Availability, Partition tolerance simultaneously.", "tags": ["CAP", "distributed-systems", "trade-offs"]},
    {"question": "How would you design a real-time chat application?", "domain": "system_design", "difficulty": "medium", "answer_key": "WebSocket connections, message queue, presence service, message storage, notification system.", "tags": ["chat", "websockets", "real-time"]},
    {"question": "What is database sharding? When and how would you implement it?", "domain": "system_design", "difficulty": "hard", "answer_key": "Horizontal partitioning across servers. Use consistent hashing, shard key selection.", "tags": ["sharding", "horizontal-scaling", "databases"]},
    {"question": "Explain microservices vs monolithic architecture. Tradeoffs?", "domain": "system_design", "difficulty": "medium", "answer_key": "Microservices: scalable, complex. Monolith: simpler, coupled. Choose based on team/scale.", "tags": ["microservices", "monolith", "architecture"]},
    {"question": "How would you design a rate limiter?", "domain": "system_design", "difficulty": "medium", "answer_key": "Token bucket, sliding window, fixed window algorithms. Redis for distributed limiting.", "tags": ["rate-limiting", "token-bucket", "throttling"]},
    {"question": "How would you design a notification system that supports email, SMS, and push notifications?", "domain": "system_design", "difficulty": "medium", "answer_key": "Fan-out service, template engine, priority queues per channel, delivery tracking, retry logic.", "tags": ["notifications", "fan-out", "messaging"]},
    {"question": "Explain eventual consistency. How do systems like DynamoDB or Cassandra achieve it?", "domain": "system_design", "difficulty": "hard", "answer_key": "Nodes may temporarily disagree. Anti-entropy, read-repair, vector clocks for conflict resolution.", "tags": ["eventual-consistency", "NoSQL", "distributed"]},

    # ======== Cloud / AWS ========
    {"question": "Explain the difference between EC2, Lambda, and ECS. When to use each?", "domain": "cloud", "difficulty": "medium", "answer_key": "EC2: full VMs. Lambda: serverless functions. ECS: container orchestration.", "tags": ["EC2", "Lambda", "ECS", "compute"]},
    {"question": "What is Infrastructure as Code? Compare Terraform and CloudFormation.", "domain": "cloud", "difficulty": "medium", "answer_key": "Declarative infrastructure management. Terraform is cloud-agnostic, CloudFormation is AWS-specific.", "tags": ["IaC", "Terraform", "CloudFormation"]},
    {"question": "Explain the shared responsibility model in AWS.", "domain": "cloud", "difficulty": "easy", "answer_key": "AWS manages infrastructure security, customer manages data/application security.", "tags": ["security", "shared-responsibility"]},
    {"question": "How would you design a highly available application on AWS?", "domain": "cloud", "difficulty": "hard", "answer_key": "Multi-AZ, auto-scaling, load balancer, RDS failover, S3 for static assets, CloudFront CDN.", "tags": ["high-availability", "multi-AZ", "auto-scaling"]},
    {"question": "What is a VPC and how do subnets, route tables, and security groups work together?", "domain": "cloud", "difficulty": "medium", "answer_key": "VPC is isolated network. Public/private subnets, route tables define traffic, security groups act as firewalls.", "tags": ["VPC", "networking", "security-groups"]},

    # ======== AI/ML ========
    {"question": "Explain the difference between supervised, unsupervised, and reinforcement learning.", "domain": "ai_ml", "difficulty": "easy", "answer_key": "Supervised: labeled data. Unsupervised: pattern discovery. RL: reward-based learning.", "tags": ["learning-types", "fundamentals"]},
    {"question": "What is the transformer architecture? Explain self-attention.", "domain": "ai_ml", "difficulty": "hard", "answer_key": "Parallel sequence processing using Q/K/V attention. Self-attention computes token relationships.", "tags": ["transformers", "attention", "NLP"]},
    {"question": "Explain the bias-variance tradeoff in machine learning.", "domain": "ai_ml", "difficulty": "medium", "answer_key": "High bias = underfitting. High variance = overfitting. Goal is optimal balance.", "tags": ["bias-variance", "model-selection"]},
    {"question": "What is transfer learning? How is it used with pre-trained models?", "domain": "ai_ml", "difficulty": "medium", "answer_key": "Reusing learned features from one task for another. Fine-tune last layers for specific domain.", "tags": ["transfer-learning", "fine-tuning"]},
    {"question": "Explain RAG (Retrieval Augmented Generation) and its benefits.", "domain": "ai_ml", "difficulty": "medium", "answer_key": "Combine retrieval from knowledge base with LLM generation. Reduces hallucination, uses current data.", "tags": ["RAG", "LLM", "knowledge-retrieval"]},
    {"question": "What is LoRA fine-tuning? How does it differ from full fine-tuning?", "domain": "ai_ml", "difficulty": "hard", "answer_key": "Low-rank adapter matrices instead of updating all weights. Much fewer parameters, preserves base model.", "tags": ["LoRA", "PEFT", "fine-tuning"]},
    {"question": "What are embeddings? How are they used in recommendation systems and search?", "domain": "ai_ml", "difficulty": "medium", "answer_key": "Dense vector representations capturing semantics. Used for similarity search, recommendations via nearest neighbor.", "tags": ["embeddings", "vector-search", "recommendations"]},
    {"question": "Explain the attention mechanism. What problem does it solve over RNNs?", "domain": "ai_ml", "difficulty": "hard", "answer_key": "Attention lets model focus on relevant parts of input. Solves long-range dependency problem of RNNs.", "tags": ["attention", "RNN", "sequence-modeling"]},

    # ======== Backend Development ========
    {"question": "Explain RESTful API design principles. What makes a good API?", "domain": "backend", "difficulty": "medium", "answer_key": "Resource-based URIs, proper HTTP methods, status codes, versioning, pagination.", "tags": ["REST", "API-design"]},
    {"question": "What is the N+1 query problem and how do you solve it?", "domain": "backend", "difficulty": "medium", "answer_key": "Loading related records one-by-one. Solve with eager loading, joins, or batch queries.", "tags": ["N+1", "ORM", "query-optimization"]},
    {"question": "Explain OAuth 2.0 and JWT. How do they work together?", "domain": "backend", "difficulty": "hard", "answer_key": "OAuth 2.0 is authorization framework. JWT is token format for stateless authentication.", "tags": ["OAuth", "JWT", "authentication"]},
    {"question": "What is CQRS? When would you use it?", "domain": "backend", "difficulty": "hard", "answer_key": "Command Query Responsibility Segregation. Separate read/write models for complex domains.", "tags": ["CQRS", "event-sourcing"]},
    {"question": "Explain database indexing. How does a B-tree index work?", "domain": "backend", "difficulty": "medium", "answer_key": "Index speeds up lookups. B-tree maintains sorted data with O(log n) search.", "tags": ["indexing", "B-tree", "databases"]},
    {"question": "What is connection pooling and why is it important for database performance?", "domain": "backend", "difficulty": "medium", "answer_key": "Reusing DB connections instead of creating new ones. Reduces overhead, prevents connection exhaustion.", "tags": ["connection-pooling", "performance", "databases"]},
    {"question": "Explain message queues. Compare RabbitMQ, Kafka, and SQS.", "domain": "backend", "difficulty": "hard", "answer_key": "Async communication. RabbitMQ: flexible routing. Kafka: high-throughput log. SQS: managed, simple.", "tags": ["message-queues", "Kafka", "RabbitMQ"]},

    # ======== Frontend ========
    {"question": "Explain the Virtual DOM in React. Why is it important?", "domain": "frontend", "difficulty": "medium", "answer_key": "In-memory DOM representation. Enables efficient diffing and batch updates to real DOM.", "tags": ["React", "virtual-DOM", "rendering"]},
    {"question": "What is the difference between SSR, CSR, and SSG?", "domain": "frontend", "difficulty": "medium", "answer_key": "SSR: server-rendered. CSR: client-rendered. SSG: pre-built at build time. Each has performance tradeoffs.", "tags": ["SSR", "CSR", "SSG", "rendering"]},
    {"question": "Explain CSS specificity and the cascade. How do you manage CSS at scale?", "domain": "frontend", "difficulty": "medium", "answer_key": "Inline > ID > Class > Element. Use CSS modules, BEM, or CSS-in-JS for scalable architecture.", "tags": ["CSS", "specificity", "architecture"]},
    {"question": "What are React hooks? Explain useState, useEffect, and useRef.", "domain": "frontend", "difficulty": "medium", "answer_key": "Hooks let functional components use state/lifecycle. useState for state, useEffect for side-effects, useRef for mutable refs.", "tags": ["React", "hooks", "state-management"]},
    {"question": "Explain web accessibility (a11y). What are ARIA roles and how do you test for accessibility?", "domain": "frontend", "difficulty": "medium", "answer_key": "Making web usable for all. ARIA provides semantic meaning. Test with axe, Lighthouse, screen readers.", "tags": ["accessibility", "ARIA", "a11y"]},

    # ======== ServiceNow ========
    {"question": "What is a ServiceNow Glide Record? How do you use it?", "domain": "servicenow", "difficulty": "easy", "answer_key": "API for database operations. GlideRecord queries, inserts, updates records in tables.", "tags": ["GlideRecord", "database"]},
    {"question": "Explain the difference between Business Rules and Client Scripts in ServiceNow.", "domain": "servicenow", "difficulty": "medium", "answer_key": "Business Rules run server-side on DB ops. Client Scripts run in browser for form interactions.", "tags": ["business-rules", "client-scripts"]},
    {"question": "What is the ServiceNow Flow Designer and how does it differ from workflows?", "domain": "servicenow", "difficulty": "medium", "answer_key": "Flow Designer is newer, no-code automation. Workflows are legacy, script-heavy.", "tags": ["Flow-Designer", "workflows", "automation"]},
    {"question": "Explain ServiceNow scoped applications. How do they enforce isolation?", "domain": "servicenow", "difficulty": "hard", "answer_key": "Scoped apps have own namespace, access controls, and tables. Prevents cross-app conflicts.", "tags": ["scoped-apps", "isolation"]},

    # ======== DevOps ========
    {"question": "Explain CI/CD pipelines. What tools have you used?", "domain": "devops", "difficulty": "medium", "answer_key": "Automated build, test, deploy. Tools: Jenkins, GitHub Actions, GitLab CI, CircleCI.", "tags": ["CI/CD", "automation", "pipelines"]},
    {"question": "What is Docker? Explain containers vs VMs.", "domain": "devops", "difficulty": "easy", "answer_key": "Containers share OS kernel (lightweight). VMs have full OS (isolated but heavy).", "tags": ["Docker", "containers", "virtualization"]},
    {"question": "Explain Kubernetes architecture. What are pods, services, and deployments?", "domain": "devops", "difficulty": "hard", "answer_key": "Pods: smallest unit. Services: networking. Deployments: declarative pod management with scaling.", "tags": ["Kubernetes", "pods", "orchestration"]},
    {"question": "What is GitOps? How does it differ from traditional deployment?", "domain": "devops", "difficulty": "medium", "answer_key": "Git as single source of truth for infrastructure. Declarative, auditable, automated reconciliation.", "tags": ["GitOps", "deployment", "infrastructure"]},

    # ======== Behavioral ========
    {"question": "Tell me about a time you had to debug a complex production issue.", "domain": "behavioral", "difficulty": "medium", "answer_key": "Look for: systematic approach, communication, root cause analysis, prevention.", "tags": ["debugging", "problem-solving"]},
    {"question": "How do you handle disagreements with team members about technical decisions?", "domain": "behavioral", "difficulty": "medium", "answer_key": "Look for: data-driven approach, empathy, compromise, focus on outcomes.", "tags": ["teamwork", "conflict-resolution"]},
    {"question": "Describe a project you're most proud of and why.", "domain": "behavioral", "difficulty": "easy", "answer_key": "Look for: passion, impact, technical depth, lessons learned.", "tags": ["passion", "impact"]},
    {"question": "Tell me about a time you had to learn a new technology quickly under pressure.", "domain": "behavioral", "difficulty": "medium", "answer_key": "Look for: learning strategy, time management, practical application, asking for help.", "tags": ["learning", "adaptability"]},
]


# ─── Resume highlight collection (per session) ─────────────────────────────
_resume_collection = None


def index_resume_highlights(profile: dict, session_id: str) -> bool:
    """
    Index a candidate's resume highlights into a dedicated ChromaDB collection.
    This lets RAG query the candidate's actual background to generate
    personalised, targeted interview questions from their own experience.

    Indexed documents include:
      - Each detected skill (with domain label)
      - Project descriptions
      - Education entries
      - Summary chunks derived from experience years

    Args:
        profile: output of resume_analyzer.analyze_resume()
        session_id: unique interview session ID (used as collection namespace)

    Returns:
        True if indexed successfully
    """
    global _resume_collection

    try:
        import chromadb
        from .embeddings import SentenceTransformerEmbeddingFunction

        # Use the global client if already initialised
        global _client
        if _client is None:
            _client = chromadb.Client()

        embedding_fn = SentenceTransformerEmbeddingFunction()
        collection_name = f"resume_{session_id[:16].replace('-', '_')}"

        _resume_collection = _client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_fn,
        )

        if _resume_collection.count() > 0:
            logger.info(f"Resume collection already indexed ({_resume_collection.count()} docs)")
            return True

        documents = []
        metadatas = []
        ids = []
        idx = 0

        skills = profile.get("skills", [])
        skill_graph = profile.get("skill_graph", {})
        projects = profile.get("projects", [])
        education = profile.get("education", [])
        exp_years = profile.get("experience_years", 0)
        name = profile.get("name", "Candidate")

        # ── Index skills as question seeds ──
        for domain, domain_skills in skill_graph.items():
            for skill in domain_skills:
                # Each skill becomes a queryable document phrased as an interview topic
                doc = (
                    f"Candidate skill: {skill}\n"
                    f"Domain: {domain}\n"
                    f"Experience level: {_exp_to_level(exp_years)}\n"
                    f"Generate interview question about {skill} for someone with {exp_years} years experience"
                )
                documents.append(doc)
                metadatas.append({
                    "type": "skill",
                    "skill": skill,
                    "domain": domain,
                    "experience_years": str(exp_years),
                    "difficulty": _exp_to_difficulty(exp_years),
                    "candidate_name": name,
                })
                ids.append(f"res_skill_{idx}")
                idx += 1

        # ── Index projects as question seeds ──
        for i, project in enumerate(projects[:8]):
            if len(project) < 10:
                continue
            doc = (
                f"Candidate project: {project}\n"
                f"Skills: {', '.join(skills[:8])}\n"
                f"Ask about this project's technical decisions and implementation"
            )
            documents.append(doc)
            metadatas.append({
                "type": "project",
                "project": project[:200],
                "domain": _infer_domain_from_text(project, skills),
                "experience_years": str(exp_years),
                "difficulty": _exp_to_difficulty(exp_years),
                "candidate_name": name,
            })
            ids.append(f"res_proj_{idx}")
            idx += 1

        # ── Index education entries ──
        for edu in education[:3]:
            if len(edu) < 5:
                continue
            doc = f"Candidate education: {edu}. Ask about theoretical concepts from this background."
            documents.append(doc)
            metadatas.append({
                "type": "education",
                "education": edu[:200],
                "domain": "general",
                "difficulty": _exp_to_difficulty(exp_years),
                "candidate_name": name,
            })
            ids.append(f"res_edu_{idx}")
            idx += 1

        # ── Index experience-level summary ──
        exp_doc = (
            f"Candidate: {name}\n"
            f"Experience: {exp_years} years — {_exp_to_level(exp_years)}\n"
            f"Key skills: {', '.join(skills[:12])}\n"
            f"Ask questions appropriate for {_exp_to_level(exp_years)} level engineer"
        )
        documents.append(exp_doc)
        metadatas.append({
            "type": "summary",
            "domain": "general",
            "difficulty": _exp_to_difficulty(exp_years),
            "experience_years": str(exp_years),
            "candidate_name": name,
        })
        ids.append(f"res_summary_{idx}")

        if documents:
            _resume_collection.add(documents=documents, metadatas=metadatas, ids=ids)
            logger.info(
                f"Resume indexed: {len(documents)} chunks for '{name}' "
                f"({exp_years} yrs, {len(skills)} skills) into '{collection_name}'"
            )

        return True

    except Exception as e:
        logger.error(f"Resume indexing failed: {e}")
        return False


def query_resume_collection(
    query: str,
    session_id: str,
    top_k: int = 3,
    domain_filter: Optional[str] = None,
) -> list:
    """
    Query the candidate's indexed resume collection to get personalised
    context for question generation.
    """
    global _resume_collection
    if _resume_collection is None:
        return []

    try:
        kwargs = {"query_texts": [query], "n_results": min(top_k, _resume_collection.count() or top_k)}
        if domain_filter:
            kwargs["where"] = {"domain": domain_filter}

        results = _resume_collection.query(**kwargs)
        hits = []
        if results and results.get("documents"):
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results.get("metadatas") else {}
                hits.append({"document": doc, **meta})
        return hits
    except Exception as e:
        logger.warning(f"Resume collection query failed: {e}")
        return []


def _exp_to_level(years: float) -> str:
    if years >= 7:
        return "senior"
    elif years >= 3:
        return "mid-level"
    elif years >= 1:
        return "junior"
    else:
        return "entry-level"


def _exp_to_difficulty(years: float) -> str:
    if years >= 6:
        return "hard"
    elif years >= 3:
        return "medium"
    else:
        return "easy"


def _infer_domain_from_text(text: str, skills: list) -> str:
    text_lower = text.lower()
    domain_keywords = {
        "python": ["python", "django", "flask", "fastapi"],
        "frontend": ["react", "vue", "angular", "html", "css", "javascript"],
        "backend": ["node", "express", "spring", "api", "rest"],
        "cloud": ["aws", "azure", "gcp", "cloud", "lambda", "ec2"],
        "devops": ["docker", "kubernetes", "ci/cd", "jenkins"],
        "ai_ml": ["machine learning", "tensorflow", "pytorch", "nlp", "model"],
        "dsa": ["algorithm", "data structure", "sort", "search"],
        "system_design": ["design", "architecture", "scale", "distributed"],
    }
    for domain, kws in domain_keywords.items():
        if any(kw in text_lower for kw in kws):
            return domain
    # Fallback: check skills
    for skill in skills:
        for domain, kws in domain_keywords.items():
            if skill.lower() in kws:
                return domain
    return "general"


def init_knowledge_base(persist_dir: str = "./chroma_db"):
    """
    Initialize ChromaDB with custom sentence-transformer embeddings
    and seed with interview questions.
    """
    global _collection, _client

    try:
        import chromadb
        from .embeddings import SentenceTransformerEmbeddingFunction

        _client = chromadb.PersistentClient(path=persist_dir)

        # Use our custom sentence-transformer embedding function
        embedding_fn = SentenceTransformerEmbeddingFunction()

        # Create or get collection WITH custom embedding function
        _collection = _client.get_or_create_collection(
            name="interview_questions",
            metadata={"description": "Technical interview question bank"},
            embedding_function=embedding_fn,
        )

        # Check if already seeded
        if _collection.count() > 0:
            logger.info(f"Knowledge base already has {_collection.count()} documents")
            return _collection

        # Seed the knowledge base
        documents = []
        metadatas = []
        ids = []

        for i, item in enumerate(KNOWLEDGE_BASE):
            # Build a rich document string for better semantic matching
            tags_str = ", ".join(item.get("tags", []))
            doc = (
                f"Domain: {item['domain']}\n"
                f"Difficulty: {item['difficulty']}\n"
                f"Tags: {tags_str}\n"
                f"Question: {item['question']}\n"
                f"Answer Key: {item['answer_key']}"
            )
            documents.append(doc)
            metadatas.append({
                "domain": item["domain"],
                "difficulty": item["difficulty"],
                "question": item["question"],
                "answer_key": item["answer_key"],
                "tags": tags_str,
            })
            ids.append(f"q_{i}")

        # Add in batches
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]

            _collection.add(
                documents=batch_docs,
                metadatas=batch_meta,
                ids=batch_ids,
            )

        logger.info(f"Knowledge base seeded with {len(documents)} questions across {len(set(item['domain'] for item in KNOWLEDGE_BASE))} domains")
        return _collection

    except ImportError as e:
        logger.error(f"ChromaDB or sentence-transformers not installed: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to initialise knowledge base: {e}")
        return None


def get_collection():
    """Get the ChromaDB collection (initialises on first call)."""
    global _collection
    if _collection is None:
        init_knowledge_base()
    return _collection


def get_question_count() -> int:
    """Return the number of questions in the knowledge base."""
    c = get_collection()
    return c.count() if c else len(KNOWLEDGE_BASE)


def get_domains() -> List[str]:
    """Return all unique domains in the knowledge base."""
    return list(set(item["domain"] for item in KNOWLEDGE_BASE))
