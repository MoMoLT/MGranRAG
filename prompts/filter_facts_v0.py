query1 = """As a reading comprehension expert, your task is to find clues from [[Primary Document Sentences]] to support filling in the blanks in [[Clozes]]. First, check if the keywords in the cloze brackets appear in the document and determine if they provide useful information.

Sometimes, the document may not directly contain the answer. For example:
[[Clozes]]
[Xiao Ming]'s [dad] is [how old] this year.

[[Primary Document Sentences]]
<DOC 0>
[0@0] Li Ming is 32 years old this year.
[0@1] Zhao Gang is 38 years old this year.
</DOC 0>

With the supplementary fact: 
[[Supplementary Facts]]
[fact_1] Xiao Ming's dad is Zhao Gang. 

You can combine this with [0@1] to find the answer. This shows the importance of integrating supplementary facts with document information.

Another example where the document doesn't provide a complete answer:

[[Clozes]]
[#1#] [seventh largest country] population in [2023].

[[Primary Document Sentences]]
<DOC 1>
[1@0] India is a country in Asia.
[1@1] It is the world's seventh largest country.
</DOC 1>
[[Supplementary Facts]]
N/A

Here, [1@1] relates to the cloze but doesn't answer the question. You have two options:
1. Consider the document worthless for this cloze.
2. Recognize that the document identifies "seventh largest country" as India, providing a foundation for further reasoning.

Please make your choice (1 or 2) and provide your answer.
"""
reply1 = """I choice 2.Based on the primary document sentences provided, we can identify that "India is the world's seventh largest country." This information provides a foundation for further reasoning. Therefore, we can conclude that the seventh largest country in 2023 is India. So, the cloze would be filled as follows:
**Cloze:** [#1#] India's population in [2023].
"""
query2 = """You are an expert in entity replacement. Your task is to replace all pronouns and ambiguous references in a sentence with explicit entity names, making the sentence understandable without context.

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
query3 = """Great! Now I'll provide another example. Please extract the information related to the [[Clozes]] from the [[Primary Document Sentences]] and combine it with the previous discussion.

[[Clozes]]
[#1#]: [Allie Goertz] is a [satirist].
[#2#]: [Allie Goertz] is a [satirist] who wrote a song about [Milhouse], who was named after [naming source].

[[Primary Document Sentences]]
<DOC 0>
[0@0] Allie Goertz is known for her satirical songs based on various pop culture topics.
[0@1] Her videos are posted on YouTube under the name of Cossbysweater.
[0@2] Subjects of her songs have included the film "The Room", the television "The Simpsons", and the game Dungeons & Dragons.
</DOC 0>

[[Supplementary Facts]]
[fact_1] Milhouse Mussolini van Houten is a fictional character featured in the animated television series "The Simpsons", and created by Matt Groening who named the character after President Richard Nixon's middle name.
[fact_2] Tim Cook became CEO in 2011.

When you tackle a cloze task, your goal is to identify sentences from the [[Primary Document Sentences]] that can support the completion of the cloze. Sometimes, even if the document doesn't provide a complete answer, it may contain partial key information.

For example:
- If a sentence mentions keywords related to the cloze and can provide partial support (such as confirming the identity, attribute, or relationship of an entity), then it is valuable.
- If a sentence does not mention any relevant concepts or cannot be used for reasoning, it is not helpful for the current cloze and should be ignored.

As an expert in reading comprehension, you have two options:
1. If you believe the document has no value and does not provide any clues → Answer N/A.
2. If you believe the document contains valuable information, even if it only supports part of the content, it is worth recording for subsequent reasoning → Please answer in the following format:
```
### Inference
A brief inference.

### Notes
[sentence X@n BEGIN]
content: The content of the original sentence.
explicit_content: The original sentence with pronouns replaced by explicit entity names, preserving structure and meaning for context-independent understanding. 
supplementary_facts: fact_Y; fact_X (Supplementary knowledge used in this clue; N/A if none)
relevant_clozes: #X#[T]; #Y#[P] (only show supported clozes; T = full answer, P = partial support)
keywords: Keywords extracted from the original sentence that are closely related to the cloze and can be used as clues for reasoning (only extract from the original sentence).
reason: A brief reason why this sentence is recorded.
[sentence X@n END]
```

# Important Notes:
- Sentences that support the [[Clozes]] should be entered into ```[sentence X@n BEGIN] xxxx [sentence X@n END]```.
- Do not output sentences of no value (even writing N/A is not allowed).
- Do not fabricate facts.
- If there are no clues at all, simply answer: N/A
"""
reply3 = """### Inference  
Sentence [0@0] directly identifies Allie Goertz as a satirist through her creation of satirical songs, fully supporting cloze #1#. Sentence [0@2] indicates that one of the subjects of her songs is "The Simpsons," which, when combined with [fact_1], suggests a possible connection to Milhouse and offers partial support for cloze #2#. Additionally, [fact_1] explicitly states that Milhouse was named after President Richard Nixon's middle name, directly supporting the naming source portion of cloze #2#.

### Notes  
[sentence 0@0 BEGIN]  
content: Allie Goertz is known for her satirical songs based on various pop culture topics.  
explicit_content: Allie Goertz is known for Allie Goertz's satirical songs based on various pop culture topics.  
supplementary_facts: N/A  
relevant_clozes: #1#[T]; #2#[P]  
keywords: Allie Goertz; satirical songs
reason: This sentence directly identifies Allie Goertz as a satirist through her creation of satirical songs, fulfilling cloze #1 fully and supporting part of #2.
[sentence 0@0 END]  

[sentence 0@2 BEGIN]  
content: Subjects of her songs have included the film "The Room", the television "The Simpsons", and the game Dungeons & Dragons.  
explicit_content: Subjects of Allie Goertz's songs have included the film "The Room", the television "The Simpsons", and the game Dungeons & Dragons.  
supplementary_facts: fact_1  
relevant_clozes: #2#[P]  
keywords: Allie Goertz; The Simpsons; songs
reason: This sentence shows that "The Simpsons" has been a subject of Allie Goertz's songs, connecting her work to the show where Milhouse appears. Fact_1 further links Milhouse’s name origin, offering contextual support for cloze #2.
[sentence 0@2 END]
"""
query4 = """Correct! Now please answer the following question:
[[Clozes]]
[#1#]: [Allie Goertz] is a [satirist] who wrote a song about [Milhouse], who was named after [naming source].

[[Primary Document Sentences]]
<DOC 1>
[1@0] Lisa Marie Simpson is a fictional character in the animated television series "The Simpsons".
</DOC 1>

[[Supplementary Facts]]
N/A
"""
reply4 = """### Inference  
The document does not provide any information related to Allie Goertz, satire, songwriting, Milhouse, or naming sources. Therefore, none of the sentences in [[Primary Document Sentences]] can be used to support or partially answer the cloze.

### Notes  
N/A
"""
query5 = """Correct! Now please answer the following question:
[[Clozes]]
[#1#]: [Yin Phyu]'s [husband].
[#2#]: The [death location] of [Yin Phyu]'s [husband].

[[Primary Document Sentences]]
<DOC 4>
[4@0]: Yin Phyu Yin Phyu Htwe was queen consort to four consecutive kings of Ava.
[4@1]: She was the chief queen of King Thado (r. 1426-1440).
[4@2]: She later became queen to King Narapati (r. 1443-1468) and King Thihathura (1469-1481), both of whom were her step-sons, and later to King Minkhaung (r. 1481-1502).
</DOC 4>

[[Supplementary Facts]]
N/A
"""
reply5 = """### Inference
The document lists Yin Phyu's husbands—King Thado, King Narapati, King Thihathura, and King Minkhaung—supporting cloze **#1#**.

There is no information about the **death location** of any husband, so **#2#** is unsupported.

### Notes
[sentence 4@1 BEGIN]  
content: She was the chief queen of King Thado (r. 1426–1440).  
explicit_content: Yin Phyu Htwe was the chief queen of King Thado (r. 1426–1440).  
supplementary_facts: N/A  
relevant_clozes: #1#[T]
keywords: Yin Phyu Htwe; King Thado;
reason: This sentence identifies one of Yin Phyu's husbands, King Thado, providing fully supports cloze #1#.
[sentence 4@1 END]  

[sentence 4@2 BEGIN]  
content: She later became queen to King Narapati (r. 1443–1468) and King Thihathura (1469–1481), both of whom were her step-sons, and later to King Minkhaung (r. 1481–1502).  
explicit_content: Yin Phyu Htwe later became queen to King Narapati (r. 1443–1468) and King Thihathura (1469–1481), both of whom were Yin Phyu Htwe's step-sons, and later to King Minkhaung (r. 1481–1502).  
supplementary_facts: N/A  
relevant_clozes: #1#[T]  
keywords: Yin Phyu Htwe; King Narapati; King Thihathura; King Minkhaung
reason: This sentence further clarifies additional husbands of Yin Phyu, providing fully supports cloze #1#.
[sentence 4@2 END]
"""
query6 = """Correct! Now please answer the following question:
[[Clozes]]
[#1#]: [Allie Goertz] is a [satirist].
[#2#]: [Milhouse] was named after [naming source].
[#3#]: [Allie Goertz] is a [satirist] who wrote a song about [Milhouse], who was named after [naming source].

[[Primary Document Sentences]]
<DOC 2>
[2@0] Later in the series, Milhouse Mussolini van Houten is revealed that Milhouse\'s middle name is "Mussolini."
[2@1] He is created by Matt Groening who named the character after President Richard Nixon\'s middle name.
</DOC 2>

[[Supplementary Facts]]
[fact_1] Milhouse is a fictional character featured in the animated television series "The Simpsons", voiced by Pamela Hayden
"""
reply6 = """### Inference  
The document provides information about the naming origin of Milhouse, connecting it to President Richard Nixon’s middle name. This supports clozes #2# and #3#. However, there is no mention of Allie Goertz or her role as a satirist, so cloze #1# is not supported.

### Notes  
[sentence 2@1 BEGIN]  
content: He is created by Matt Groening who named the character after President Richard Nixon's middle name.  
explicit_content: Milhouse is created by Matt Groening who named the character after President Richard Nixon's middle name.  
supplementary_facts: fact_1  
relevant_clozes: #2#[T]; #3#[P]  
keywords: Milhouse; named after; Richard Nixon; middle name  
reason: Explicitly states Milhouse's naming source, fully answering cloze #2 and partially supporting cloze #3.
[sentence 2@1 END]
"""
query7 = """Correct! You're truly an excellent reading comprehension master! Now please answer the following question:
[[Clozes]]
[#1#]: [world's seventh-largest country] is located in [continent].
[#2#]: [genus of moth] contains [one species].
[#3#]: The [genus of moth] found only in [world's seventh-largest country] contains [one species].

[[Primary Document Sentences]]
<DOC 3>
[3@0]: India, officially the Republic of India ("Bhārat Gaṇarājya"), is a country in South Asia.
[3@1]: It is the seventh-largest country by area, the second-most populous country (with over 1.2 billion people), and the most populous democracy in the world.
[3@2]: It is bounded by the Indian Ocean on the south, the Arabian Sea on the southwest, and the Bay of Bengal on the southeast.
[3@3]: It shares land borders with Pakistan to the west; China, Nepal, and Bhutan to the northeast; and Myanmar (Burma) and Bangladesh to the east.
</DOC 3>

[[Supplementary Facts]]
N/A
"""
reply7 = """### Inference  
Sentence [3@0] identifies India as a country located in South Asia, which supports determining the continent for cloze #1#. Sentence [3@1] explicitly states that India is the seventh-largest country by area, directly supporting cloze #1#. For cloze #3#, although the document mentions India and its geographical and political neighbors, it does not provide any information about a genus of moth or species found in India, so no part of the document supports clozes #2# or the moth-related portion of #3#.  

### Notes  
[sentence 3@0 BEGIN]  
content: India, officially the Republic of India ("Bhārat Gaṇarājya"), is a country in South Asia.  
explicit_content: India, officially the Republic of India ("Bhārat Gaṇarājya"), is a country in South Asia.  
supplementary_facts: N/A  
relevant_clozes: #1#[P]  
keywords: India; South Asia  
reason: This sentence provides the location (continent) of India, which supports answering the second blank in cloze #1#.  
[sentence 3@0 END]  

[sentence 3@1 BEGIN]  
content: It is the seventh-largest country by area, the second-most populous country (with over 1.2 billion people), and the most populous democracy in the world.  
explicit_content: India is the seventh-largest country by area, the second-most populous country (with over 1.2 billion people), and the most populous democracy in the world.  
supplementary_facts: N/A  
relevant_clozes: #1#[T]  
keywords: seventh-largest country; India
reason: This sentence directly identifies India as the seventh-largest country, fully supporting cloze #1#.  
[sentence 3@1 END]
"""
question_input = """Correct! Now please answer the following question:
[[Clozes]]
${clozes}

[[Primary Document Sentences]]
${document}

[[Supplementary Facts]]
${facts}

/no_think
"""
query8 = """[[Clozes]]
[#1#]: [Saw Thanda]'s [husband].
[#2#]: The [death location] of [Saw Thanda]'s [husband].
[#3#]: Where did Saw Thanda's husband die?

[[Primary Document Sentences]]
<DOC 4>
[4@0]: Saw Thanda Saw Thanda Dewi was queen consort to four consecutive of kings of Arakan.
[4@1]: She was the chief queen of King Dikkha (r. 1554–56).
[4@2]: She later became queen to King Saw Hla (r. 1556–64) and King Sekkya (1564–72), both of whom were her step-sons, and later to King Phalaung (r. 1572–93).
</DOC 4>

[[Supplementary Facts]]
N/A
"""
query9 = """Correct! You're truly an excellent reading comprehension master! Now please answer the following question:
[[Clozes]]
[#1#]: Which film has the director who is older, God'S Gift To Women or Aldri Annet Enn Bråk?

[[Primary Document Sentences]]
<DOC 3>
[3@0]: Aldri annet enn bråk Aldri annet enn bråk is a 1954 Norwegian comedy- drama film, edited by Edith Carlmar.
[3@1]: It takes place in a working- class setting in the city of Oslo. The English title is" Nothing but trouble".
[3@2]: A 1955 Danish remake," Altid ballade", was directed by Gabriel Axel.
</DOC 3>

[[Supplementary Facts]]
N/A
"""
prompt_template = [
    {"role": "user", "content": query1},
    {"role": "assistant", "content": reply1},
    {"role": "user", "content": query2},
    {"role": "assistant", "content": reply2},
    {"role": "user", "content": query3},
    {"role": "assistant", "content": reply3},
    {"role": "user", "content": query4},
    {"role": "assistant", "content": reply4},
    {"role": "user", "content": query5},
    {"role": "assistant", "content": reply5},
    # {"role": "user", "content": query6},
    # {"role": "assistant", "content": reply6},
    {"role": "user", "content": query7},
    {"role": "assistant", "content": reply7},
    {"role": "user", "content": question_input},
]