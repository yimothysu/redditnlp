"""Module for extracting topics from text content using BERTopic model."""

from bertopic import BERTopic

def extract_topics(texts, num_topics=10):
    """Extract topics from a list of texts using BERTopic.
    
    Uses BERTopic to identify main themes and topics in the text corpus.
    Preprocesses texts to remove stopwords and applies topic modeling.
    
    Args:
        texts: List of text documents for topic modeling
        num_topics: Number of topics to extract (default: 10)
        
    Returns:
        dict: Dictionary containing:
            - topics: List of topic numbers assigned to each document
            - topic_info: DataFrame with information about each topic
            - topic_representations: Dict mapping topic IDs to word-weight pairs
            - error: Error message if processing fails
    """
    try:
        # Preprocess texts to remove stopwords
        preprocessed_texts = []
        for text in texts:
            preprocessed_text = preprocess_text_for_n_grams(text)
            if preprocessed_text.strip():  # Only add non-empty texts
                preprocessed_texts.append(preprocessed_text)
        
        # Skip if there are not enough texts after preprocessing
        if len(preprocessed_texts) < 5:
            return {
                "error": "Not enough text documents after preprocessing",
                "topics": [],
                "topic_info": [],
                "topic_representations": {}
            }
        
        # Initialize and fit BERTopic model
        topic_model = BERTopic()
        topics, probs = topic_model.fit_transform(preprocessed_texts)
        
        # Get topic info and representation
        topic_info = topic_model.get_topic_info()
        
        # Get the topic representations (words and their weights)
        topic_representations = {}
        for topic_id in set(topics):
            if topic_id != -1:  # Skip outlier topic
                topic_words = topic_model.get_topic(topic_id)
                topic_representations[str(topic_id)] = topic_words
        
        return {
            "topics": topics,
            "topic_info": topic_info.to_dict('records'),
            "topic_representations": topic_representations
        }
    except Exception as e:
        print(f"Error in topic extraction: {str(e)}")
        return {
            "error": str(e),
            "topics": [],
            "topic_info": [],
            "topic_representations": {}
        }