from guidance import models, gen
import guidance
lm = models.Transformers('mistralai/Mistral-7B-v0.1')

@guidance(stateless=True)
def ner_instruction(lm, input):
    lm += f'''\
    Please tag PII words with PII_NAME, PII_ADDRESS, PII_EMAIL, PII_SSN, PII_PASSPORT, PII_DRV_LICENSE, PII_CC, PII_DOB, PII_PHONE, PII_LOGIN, PII_MED_REC, PII_HEALTH_INS, PII_IP_ADDR, PII_BANK_ACC, PII_EMPLOYMENT, PII_EDUCATION, PII_VEHICLE_REG, PII_PHOTO, PII_MAIDEN_NAME or nothing
    ---
    Input: John worked at Apple. His ID is 10000235 and his DOB is 12/1/1948
    Output:
    John: PII_NAME
    10000235: PII_EMPLOYMENT
    12/1/1948: PII_DOB
    ---
    Input: My credit card number is 4111-1111-1111-1111 and my driver's license is ABC123XYZ.
    Output:
    4111-1111-1111-1111: PII_CC
    ABC123XYZ: PII_DRV_LICENSE
    ---
    Input: I graduated from Oxford University, and my student ID was OX12345.
    Output:
    Oxford University: PII_EDUCATION
    OX12345: PII_EDUCATION
    ---
    Input: Contact me at 555-1234 or jane.doe@email.com.
    Output:
    555-1234: PII_PHONE
    jane.doe@email.com: PII_EMAIL
    ---
    Input: {input}
    Output:
    '''
    return lm

input = "My name is John Doe and I live at 123 Maple Street, Springfield. You can contact me at johndoe@example.com. My social security number is 123-45-6789."
print(lm + ner_instruction(input) + gen(stop='---'))
