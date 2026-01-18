# -----------------------------
# CNN + RAG Smart City Prototype
# -----------------------------

# STEP 1: CNN MODEL (Simulated)
def cnn_predict(image_path):
    """
    Simulated CNN output
    """
    age_group = "20–35"
    gender_distribution = {
        "female": 55,
        "male": 45
    }
    return age_group, gender_distribution


# STEP 2: QUERY BUILDER
def build_query(age_group, gender_distribution, location):
    query = f"""
    Location: {location}
    Age group detected: {age_group}
    Gender distribution: {gender_distribution}
    Generate smart city planning insights aligned with SDG 11.
    """
    return query


# STEP 3: RETRIEVER (Simple RAG Retrieval)
def retrieve_documents(query, knowledge_file="knowledge_base.txt"):
    with open(knowledge_file, "r") as f:
        documents = f.readlines()

    # Simple keyword-based retrieval (prototype purpose)
    retrieved_docs = []
    for doc in documents:
        if "transport" in doc.lower() or "urban" in doc.lower():
            retrieved_docs.append(doc.strip())

    return retrieved_docs


# STEP 4: GENERATOR (RAG OUTPUT)
def generate_rag_response(query, retrieved_docs):
    response = (
        "Based on SDG 11 guidelines and urban mobility studies, "
        "the detected concentration of young adults at the metro station "
        "indicates a strong demand for frequent public transportation services. "
        "The retrieved documents emphasize the need for enhanced safety measures "
        "and employment-oriented infrastructure near transit hubs to support "
        "sustainable and inclusive urban development."
    )
    return response


# STEP 5: MAIN PIPELINE
def run_rag_pipeline(image_path, location):
    # CNN prediction
    age_group, gender_distribution = cnn_predict(image_path)

    # Build query
    query = build_query(age_group, gender_distribution, location)

    # Retrieve documents
    retrieved_docs = retrieve_documents(query)

    # Generate final insight
    final_output = generate_rag_response(query, retrieved_docs)

    # Display results
    print("---- CNN OUTPUT ----")
    print("Age Group:", age_group)
    print("Gender Distribution:", gender_distribution)

    print("\n---- RETRIEVED DOCUMENTS ----")
    for doc in retrieved_docs:
        print("-", doc)

    print("\n---- RAG GENERATED INSIGHT ----")
    print(final_output)


# RUN PROTOTYPE
if __name__ == "__main__":
    run_rag_pipeline(
        image_path="sample_image.jpg",
        location="Metro Station"
    )
