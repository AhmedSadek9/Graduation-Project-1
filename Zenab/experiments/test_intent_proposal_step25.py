"""Run: python experiments/test_intent_proposal_step25.py"""
from intent_rules_step25 import proposed_allowed_fact_types as intent

def test(question, expected):
    actual,reason=intent(question)
    assert actual==expected,(question,actual,reason,expected)
    print('PASS',question,'=>',reason)

def main():
    test('What did Adam and I make at the picnic?',{'shared_memory'})
    test('What did Rania and I exchange at a family gathering?',{'shared_memory'})
    test('Where did Mostafa and I go on a family trip?',{'shared_memory'})
    test('What collection does Mostafa keep?',{'personal_interest'})
    test('What did Nada do at a family gathering?',{'shared_memory'})
    test("What is Omar Hassan's phone number?",set())
    test('Who is this person?',{'identity_relationship'})
    test('What can I talk to Salma Nabil about?',{'conversation_starter'})
    test('Explain the weather forecast',None)
    print('STEP 25 RULE TESTS: 9/9 PASSED')
if __name__=='__main__': main()
