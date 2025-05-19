from fastcoref import FCoref
from fastcoref import spacy_component
import spacy # type: ignore
import nltk # type: ignore
import queue
import re

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("fastcoref")
nltk.download('punkt')

seperator = " ==== " # divides the context of a comment and the body of a comment 

'''
    Recursive helper function for set_context_for_post_comments 
'''
def set_context_for_comment(comment, context):
    comment.context = context 
    for i in range(len(comment.replies)):
        comment.replies[i] = set_context_for_comment(comment.replies[i], comment.context + " " + comment.text)
    return comment 


'''
    The context for a top level comment is just the post's title and description 
    The context for any nested comment is the post's title and description + the parent comments 

    context DOESN'T include the comment's text, so there's no need to add a seperator in context 
'''
def set_context_for_post_comments(post):
    # resolve the pronouns in post.title and post.description 
    post_title_and_description = post.title + seperator + post.description
    try:
        doc = nlp(post_title_and_description, component_cfg={"fastcoref": {'resolve_text': True}})
        if doc is not None:
            post_title_and_description = doc._.resolved_text
            seperator_idx = post_title_and_description.find(seperator) # the idx that the seperator STARTS 
            post.title = post_title_and_description[:seperator_idx]
            post.description = post_title_and_description[seperator_idx + len(seperator):]
    except Exception as e:
        print(f"[FATAL] Failed during NLP pipeline execution for post title & description.")
        print(f"Error: {e}")

    # set the context field for all the post's comments 
    top_level_comment_context = post.title + " " + post.description 
    for i in range(len(post.top_level_comments)):
        post.top_level_comments[i] = set_context_for_comment(post.top_level_comments[i], top_level_comment_context)
    return post 


def resolve_pronouns_for_post(post):
    post = set_context_for_post_comments(post)
    resolved_comments = []
    flattened_context_plus_comments = get_flattened_context_plus_comments_bfs(post)
    try:
        docs = list(nlp.pipe(flattened_context_plus_comments, component_cfg={"fastcoref": {'resolve_text': True}}))
        for comment, doc in zip(flattened_context_plus_comments, docs):
            try:
                if doc is not None and doc._.resolved_text:
                    # weird bug during coreference resolution process where fastcoref makes an already 
                    # possessive entities possessive --> Ex: Musk's's
                    if re.search(r"\b\w+'s's\b", doc._.resolved_text):
                        print("Found a word ending in 's's!")
                        # Replace all occurrences of "'s's" at the end of a word with "'s"
                        resolved_comments.append(re.sub(r"'s's\b", "'s", doc.resolved_text))
                    elif re.search(r"\b\w+’s’s\b", doc._.resolved_text):
                        print("Found a word ending in ’s’s!")
                        # Replace all occurrences of "’s’s" at the end of a word with "’s"
                        resolved_comments.append(re.sub(r"’s’s\b", "’s", doc.resolved_text))
                    else:
                        resolved_comments.append(doc._.resolved_text)
                    print("successfully resolved comment! ^~^")
                else:
                    resolved_comments.append(comment)
            except Exception as e:
                print(f"[WARNING] Failed to resolve pronouns for comment")
                print(f"Error: {e}")
                resolved_comments.append(comment)
    except Exception as e:
        print(f"[FATAL] Failed during NLP pipeline execution for post comments.")
        print(f"Error: {e}")
        resolved_comments = flattened_context_plus_comments.copy()
    
    if len(resolved_comments) != len(flattened_context_plus_comments):
        raise ValueError(f"Mismatch in comment count: len(comments)={len(flattened_context_plus_comments)}, len(resolved_comments)={len(resolved_comments)}")
    
    # remove the context and seperator from each resolved comment in resolved_comments 
    for i in range(len(resolved_comments)):
        if seperator in resolved_comments[i]:
            seperator_idx = resolved_comments[i].find(seperator) # the idx that the seperator STARTS 
            resolved_comments[i] = resolved_comments[i][seperator_idx + len(seperator):].strip() 

    unvisited = queue.Queue()
    for x in range(len(post.top_level_comments)):
        unvisited.put((x,))
    while not unvisited.empty():
        indices = unvisited.get() 
        if len(indices) == 1:
            post.top_level_comments[indices[0]].text = resolved_comments.pop(0)
            replies = post.top_level_comments[indices[0]].replies 
        else:
            # Navigate to the parent level
            parent_comment = post.top_level_comments[indices[0]]
            for index in indices[1:-1]:
                parent_comment = parent_comment.replies[index]
            parent_comment.replies[indices[-1]].text = resolved_comments.pop(0)
            replies = parent_comment.replies[indices[-1]].replies 
        for i in range(len(replies)):
            unvisited.put(indices + (i,))
    return post 

'''
    Goes through the comments of a post in breadth first search style 
    Includes the comment's context followed by the comment's text 
    Each element in flattened_context_plus_comments_bfs follows the format: [context] [seperator] [comment] 
'''
def get_flattened_context_plus_comments_bfs(post):
    flattened_context_plus_comments_bfs = []
    unvisited = queue.Queue()
    for top_level_comment in post.top_level_comments:
        unvisited.put(top_level_comment)
    while not unvisited.empty():
        comment = unvisited.get()
        flattened_context_plus_comments_bfs.append(comment.context + seperator + comment.text)
        replies = comment.replies
        for reply in replies:
            unvisited.put(reply)
    return flattened_context_plus_comments_bfs 
