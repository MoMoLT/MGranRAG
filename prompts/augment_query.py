query1 = """
Answer questions following the given format.

# Knowledge
{Example_Knowledge} 
# Question
Are It Might Get Loud and Mr. Big both Canadian documentaries? 
Let’s think step by step.
# Output
/no_think
"""
reply1 = """Mr. Big is a 2007 documentary which examines the "Mr. Big" undercover methods used by the Royal Canadian Mounted Police. However, It Might Get Loud is a 2008 American documentary film. So the answer is no."""

query2 = """# Knowledge
{Example_Knowledge}
# Question
Were László Benedek and Leslie H. Martinson both film directors? 
Let’s think step by step. 
# Output
/no_think
"""
reply2 = """László Benedek was a Hungarian-born film director and Leslie H. Martinson was an American film director. So the answer is yes."""

query3 = """# Knowledge
{Example_Knowledge}
# Question
Lucium was confimed to be an impure sample of yttrium by an English chemist who became the president of what? 
Let’s think step by step.
# Output
/no_think
"""
reply3 = """Lucium was confimed to be an impure sample of yttrium by William Crookes. William Crookes is Sir William Crookes. Sir William Crookes became the president of the Society for Psychical Research. So the answer is Society for Psychical Research."""

query4 = """# Knowledge
${knowledges}
# Question
${question}
Let’s think step by step.
# Output
/no_think
"""




prompt_template = [
    {"role": "user", "content": query1},
    {"role": "assistant", "content": reply1},
    {"role": "user", "content": query2},
    {"role": "assistant", "content": reply2},
    {"role": "user", "content": query3},
    {"role": "assistant", "content": reply3},
    {"role": "user", "content": query4},
]