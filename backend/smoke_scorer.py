"""Isolated smoke test of the scorer machinery (no opf load): does the adversary+judge recover a KNOWN answer?"""

import score

m = score.LunaModel()
label = "the two companies / parties to this M&A deal"

print("check_correctness exact :", score.check_correctness("Adobe", ["Adobe"], m))
print("check_correctness close :", score.check_correctness("Figma", ["Figma Inc"], m))
print("check_correctness wrong :", score.check_correctness("Adobe", ["Tesla"], m))

txt = ("intake: [ORG_1] agreed to acquire [ORG_2] for $20 billion in cash and stock; "
       "[ORG_2] makes cloud-based collaborative interface design software; announced 2022.")
guesses = score._adversary_infer(txt, [label], m)
print("adversary guesses       :", guesses)
top = guesses.get(label.lower(), [])
print("scored recovery of Adobe:", max(score.check_correctness("Adobe", top, m)) if top else 0.0)
print("scored recovery of Figma:", max(score.check_correctness("Figma", top, m)) if top else 0.0)
