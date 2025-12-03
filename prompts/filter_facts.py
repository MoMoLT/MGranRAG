query1 = """As a reading comprehension expert, retrieve clues from Primary Document Sentences to address Questions. First, verify if question keywords exist in the document and whether they offer valuable information.

If the document has no direct answers, integrate Supplementary Facts (e.g., Question: How old is Xiao Ming's dad?; Supplementary Fact: Xiao Ming's dad is Zhao Gang; Document: Zhao Gang is 38 → Answer: 38).

For incomplete document information (e.g., Question: What is the population of the world's seventh largest country in 2023?; Document: India is the world's seventh largest country), select 1 (document is useless) or 2 (provides a foundation for reasoning) and respond.
"""
reply1 = """I select 2. The document specifies "India is the world's seventh largest country," which serves as a key foundation for reasoning. This confirms the country in the question is India, and the population inquiry focuses on India.
Question Answer Foundation: The core subject (the world's seventh largest country) is identified as India through the document."
"""
query2 = """You are an expert in entity replacement. Your task is to replace all pronouns and ambiguous references in a sentence with explicit entity names, ensuring the sentence is understandable without context.

Examples:
- content: She is known for her songs.
  explicit_content: Allie Goertz is known for Allie Goertz's songs.
- content: He founded the company in 1976.
  explicit_content: Steve Jobs founded Apple Inc. in 1976.
- content: Her latest book was published last year.
  explicit_content: J.K. Rowling's latest book was published last year.

Output format:
explicit_content: [your resolved sentence here]

Now, here’s the document to resolve:
<DOC 0>
[0@0] Emily Rose is a renowned video game designer.
[0@1] Themes of her games have included cyberpunk societies.
</DOC 0>

content: Themes of her games have included cyberpunk societies.
explicit_content:(To be filled in)
"""
reply2 = """Themes of Emily Rose's games have included cyberpunk societies."""
query3 = """Great! Your task is to identify and extract information relevant to [[Questions]] from [[Primary Document Sentences]], then integrate it with the previous discussion context.

# Questions
Q1: What is Allie Goertz's occupation?
Q2: Allie Goertz wrote a song about Milhouse. What was Milhouse named after?

# Primary Document Sentences
<DOC 0>
[0@0] Allie Goertz is known for her satirical songs based on various pop culture topics.
[0@1] Her videos are posted on YouTube under the name of Cossbysweater.
[0@2] Subjects of her songs have included the film "The Room", the television "The Simpsons", and the game Dungeons & Dragons.
</DOC 0>

# Supplementary Facts
[fact1] Milhouse Mussolini van Houten is a fictional character featured in the animated television series "The Simpsons", and created by Matt Groening who named the character after President Richard Nixon's middle name.
[fact2] Tim Cook became CEO in 2011.

# Core Rules
1. A document sentence is valuable if it mentions question-related keywords and provides support (confirming entity identity, attribute, relationship, or laying a reasoning foundation); ignore sentences with no relevant concepts or that cannot be used for reasoning.
2. Full support (T): The sentence independently provides a specific, clear answer to the question’s core.
3. Partial support (P): The sentence offers key relevant clues but not a complete, specific answer.

# Your Options
1. No valuable clues in the document → Answer N/A.
2. Document contains valuable information (even partial support) → Answer in the following format:
```
### Inference
A brief inference linking document information to the questions.

### Notes
[sentence X@n]
explicit_content: The sentence with pronouns replaced by explicit entities (ensuring understanding without context, preserving structure/meaning).
supplementary_facts: factY; factX (supplementary knowledge used; N/A if none)
relevant_clozes: QX[T]; QY[P] (only show supported clozes; T = full answer, P = partial support; N/A if none)
keywords: Closely related keywords from the original sentence (clue for reasoning).
reason: Brief explanation of why the sentence is valuable.
[END]
```

# Important Notes
- Only include valuable document sentences in the `[sentence X@n] xxxx [END]` block.
- Do not output useless sentences (even N/A is not allowed).
- Do not fabricate facts.
- If no clues at all, simply answer: N/A
"""
reply3 = """### Inference
Document sentences confirm Allie Goertz’s connection to satirical works and "The Simpsons", which links to Milhouse via supplementary facts for Q2 reasoning.

### Notes
[sentence 0@0]
explicit_content: Allie Goertz is known for Allie Goertz's satirical songs based on various pop culture topics.
supplementary_facts: N/A
relevant_questions: Q1[P]
keywords: Allie Goertz; satirical songs
reason: Confirms Allie Goertz creates satirical works, providing partial support for her occupation in Q1.
[END]

[sentence 0@2]
explicit_content: Subjects of Allie Goertz's songs have included the film "The Room", the television "The Simpsons", and the game Dungeons & Dragons.
supplementary_facts: fact1
relevant_questions: Q2[P]
keywords: Allie Goertz; songs; The Simpsons
reason: Links Allie Goertz’s songs to "The Simpsons", and combines with fact_1 to confirm Milhouse’s connection, laying a foundation for Q2.
[END]
"""
query4 = """Correct! Now, recall the 3 Core Rules and reorganize your response to address the following question:
# Core Rules
1. A document sentence is valuable if it ...
2. Full support ...
3. Partial support ... 

# Questions
Q1: Allie Goertz is a satirist who wrote a song about Milhouse. What was Milhouse named after?

# Primary Document Sentences
<DOC 1>
[1@0] Lisa Marie Simpson is a fictional character in the animated television series "The Simpsons".
</DOC 1>

# Supplementary Facts
N/A
"""
reply4 = """### Inference  
Sentence [1@0] only mentions "Lisa Marie Simpson" and "The Simpsons", with no keywords related to the question (Allie Goertz, Milhouse, named after). It provides no support for reasoning about Milhouse's naming source, so it is irrelevant.

### Notes  
N/A
"""
query5 = """Correct! Recall the 3 Core Rules and address the following question:
# Questions
Q1: Who is Yin Phyu's husband?
Q2: Where is Yin Phyu's husband's place of death?

# Primary Document Sentences
<DOC 4>
[4@0]: Yin Phyu Htwe was queen consort to four consecutive kings of Ava.
[4@1]: She was the chief queen of King Thado (r. 1426-1440).
[4@2]: She later became queen to King Narapati (r. 1443-1468) and King Thihathura (1469-1481), both of whom were her step-sons, and later to King Minkhaung (r. 1481-1502).
</DOC 4>

# Supplementary Facts
N/A
"""
reply5 = """### Inference
Document sentences support Q1 (Yin Phyu's husband) with partial and full support; no clues back Q2 (death location).

### Notes
[sentence 4@0]
explicit_content: Yin Phyu Htwe was queen consort to four consecutive kings of Ava.
supplementary_facts: N/A
relevant_questions: Q1[P]
keywords: Yin Phyu Htwe; queen consort; four consecutive kings of Ava
reason: Offers key scope clues but no specific names of husbands → partial support.
[END]

[sentence 4@1]
explicit_content: Yin Phyu Htwe was the chief queen of King Thado (r. 1426-1440).
supplementary_facts: N/A
relevant_questions: Q1[T]
keywords: Yin Phyu Htwe; chief queen; King Thado
reason: Independently provides a specific husband’s name → full support.
[END]

[sentence 4@2]
explicit_content: Yin Phyu Htwe later became queen to King Narapati (r. 1443-1468) and King Thihathura (1469-1481), both of whom were Yin Phyu Htwe’s step-sons, and later to King Minkhaung (r. 1481-1502).
supplementary_facts: N/A
relevant_questions: Q1[T]
keywords: Yin Phyu Htwe; queen to; King Narapati; King Thihathura; King Minkhaung
reason: Independently lists three specific husbands’ names → full support.
[END]
"""
question_input = """Correct! Recall the 3 Core Rules and address the following question:
# Questions
${clozes}

# Primary Document Sentences
${document}

# Supplementary Facts
${facts}

/no_think
"""
prompt_template = [
    {"role": "user", "content": query1},
    {"role": "assistant", "content": reply1},
    # {"role": "user", "content": query2},
    # {"role": "assistant", "content": reply2},
    {"role": "user", "content": query3},
    {"role": "assistant", "content": reply3},
    {"role": "user", "content": query4},
    {"role": "assistant", "content": reply4},
    {"role": "user", "content": query5},
    {"role": "assistant", "content": reply5},
    {"role": "user", "content": question_input},
]
