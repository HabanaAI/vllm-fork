import base64
import os
from openai import OpenAI
from typing import List
import multiprocessing
from functools import partial
import argparse
import time

output_folder = "./output/"

# Set up OpenAI client to connect to vLLM server
os.environ["OPENAI_API_KEY"] = "EMPTY"
os.environ['OPENAI_API_BASE'] = "http://localhost:12345/v1/"
openai_api_key = os.environ['OPENAI_API_KEY']
openai_api_base = os.environ['OPENAI_API_BASE']
# openai_api_base = "http://localhost:8080/v1/"
# openai_api_base = "http://10.241.66.6/v1"
# openai_api_base = "http://0.0.0.0:8080/v1/"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)



system_prompt_member = """"system_prompt": {
                "objective": "Extract explicitly labeled Member-related details strictly from sections labeled 'Member', 'Patient', 'Enrollee', or 'Subscriber'. NEVER extract data from sections labeled Provider, Facility, Prescriber, or similar sections. Ensure accuracy, determinism, and avoid hallucinations. Implement intelligent spell-correction techniques to accurately handle and minimize errors from handwritten and pixelated printed texts.",
                "allowed_sections": ["Member", "Patient", "Enrollee", "Subscriber"],
                "prohibited_sections": ["Provider", "Facility", "Prescriber", "Referring Provider", "Servicing Provider", "Service Provider", "Rendering Provider", "Facility Provider", "Physician"],
                "extraction_rules": [
                    "Only extract fields explicitly labeled as Member details.",
                    "Never map or substitute Provider, Facility, or Prescriber data for Member fields.",
                    "Strictly prioritize extraction in the following order: Member > Patient > Enrollee > Subscriber. Only proceed to lower-priority sections if the higher-priority section is completely missing or explicitly empty.",
                    "Extract Member ID accurately; cross-validate labels to extract only legitimate Member IDs and avoid irrelevant IDs such as Medicaid ID or Member Plan ID.",
                    "Ensure Member First Name, Last Name, and Full Name are correctly identified; never swap or interchange these fields. Verify labels explicitly before assignment.",
                    "Implement intelligent spell-correction and character-recognition improvements specifically for handwritten and pixelated printed content.",
                    "If explicitly labeled Member fields are missing, use Subscriber-labeled fields only as fallback.",
                    "If both Member and Subscriber labeled details are absent, return empty fields.",
                    "Maintain consistent extraction results on every execution."
                ],
                "fields": {
                    "member_id": ["Member ID", "Insurance ID#", "HCID Number", "HIC", "HIC Number", "Health Plan ID Number", "ID #", "ID NO", "ID No", "ID Number", "Identification Number", "Identification Number: (see ID card)", "Insured ID", "Insured ID Number", "Member ID Number", "Members ID Number", "Membership ID Number", "Subscriber ID", "Patient ID", "Patient ID #"],
                    "medicaid_id": ["Medicaid ID", "Medicaid Number"],
                    "member_full_name": ["Full Name", "Member Name", "Patient Name", "Subscriber Name"],
                    "member_first_name": ["First Name", "Given Name"],
                    "member_last_name": ["Last Name", "Surname"],
                    "member_date_of_birth": ["Date of Birth", "DOB", "Birth Date"],
                    "member_gender": ["Gender", "Sex"],
                    "member_group_id": ["Group Number", "Policy Group Number"],
                    "member_address_line1": ["Address", "Address Line 1", "Residence Address", "Mailing Address"],
                    "member_city": ["City", "Residence City", "Mailing City"],
                    "member_state": ["State", "Residence State", "Mailing State"],
                    "member_zipcode": ["ZIP", "ZIP Code", "Postal Code"]
                },
                "bounding_box": {
                    "instruction": "Provide precise bounding box coordinates matching the original document resolution for extracted text. For missing fields, explicitly return coordinates as (0,0,0,0)."
                },
                "output_format": "Always return output as an array of dictionaries, each containing keys: 'key', 'value', and 'boundingBox'. Confidence scores should NOT be included."
            }"""
user_prompt_member = """"user_prompt": {
                "task": "Extract ONLY explicitly labeled Member details from the provided document. Strictly prioritize extraction in the order: Member > Patient > Enrollee > Subscriber. Only proceed to lower-priority sections if higher-priority sections are explicitly missing. NEVER map or infer Provider, Facility, or Prescriber data into Member fields. Ensure accuracy for Member ID, extracting only legitimate Member IDs without mistakenly extracting Medicaid ID, Member Plan ID, or any irrelevant ID. Clearly distinguish and correctly assign First Name, Last Name, and Full Name without swapping. Fields such as Medicaid ID, Address, City, State, and ZIP code are performing well▒~@~Tmaintain their current extraction accuracy without introducing any errors. If neither Member nor Subscriber details are explicitly labeled, return fields as empty with zero bounding box.",
                "fields_to_extract": [
                    "member_id", "medicaid_id", "member_full_name", "member_first_name", "member_last_name",
                    "member_date_of_birth", "member_gender", "member_group_id", "member_address_line1",
                    "member_city", "member_state", "member_zipcode"
                ],
                "extraction_conditions": {
                    "strictness": "Absolutely NO inference or hallucination. Extract ONLY explicitly labeled data from permitted sections.",
                    "consistency": "Identical output structure and extracted values in repeated executions.",
                    "text_handling": "Accurate extraction of both printed and handwritten texts."
                },
                "bounding_box": {
                    "instruction": "Provide exact integer bounding box coordinates matching original document resolution. If field is empty, explicitly set bounding box to (0,0,0,0)."
                },
                "output_format": {
                    "json_structure": [
                        {
                            "key": "<field_name>",
                            "value": "<Extracted value or empty>",
                            "boundingBox": {
                                "topLeftX": "<integer>",
                                "topLeftY": "<integer>",
                                "bottomRightX": "<integer>",
                                "bottomRightY": "<integer>"
                            }
                        }
                    ]
                }
            }"""


system_prompt_diagnose_code = """
{
    "system_prompt": {
        "objective": "Accurately locate, recognize, and extract only valid diagnosis codes (ICD-10 and DSM-5) from AUMI Behavioral Health documents. The documents may contain typed, scanned, or handwritten medical information. Do not extract other types of codes or data.",
        "allowed_labels": [
            "ICD-10 Diagnoses",
            "ICD-10 Code",
            "Dx Code",
            "Diagnosis Code",
            "Primary Diagnosis Code",
            "Principal Diagnosis Code",
            "ICD Code",
            "DSM-5 Diagnoses",
            "Behavioral Health Diagnoses"
        ],
        "prohibited_codes": [
            "Procedure codes",
            "Revenue codes",
            "CPT codes",
            "HCPCS codes",
            "Medications",
            "Treatment codes"
        ],
        "extraction_rules": [
            "Extract codes from both inline text/handwritten fields and tabular data.",
            "If a table column has an allowed label, extract the entire column.",
            "Preserve the exact format, including casing, punctuation, and spacing.",
            "Handwritten codes must be interpreted via OCR and preserved exactly.",
            "Return precise bounding box coordinates for each code on a 1000x1000 pixel grid."
        ],
        "output_format": {
            "json_structure": [
                {
                    "key": "diagnosis_code",
                    "value": "<Extracted code or empty>",
                    "boundingBox": {
                        "topLeftX": "<integer>",
                        "topLeftY": "<integer>",
                        "bottomRightX": "<integer>",
                        "bottomRightY": "<integer>"
                    }
                }
            ],
            "multiple_codes_format": "Multiple codes should be combined into a single string, separated by commas."
        },
        "step-by-step_reasoning": [
            "1. Locate a relevant label from the allowed_labels list.",
            "2. Verify that the associated code is a valid ICD-10 or DSM-5 format.",
            "3. Confirm that the format is preserved exactly as it appears.",
            "4. Extract the code and its bounding box coordinates."
        ]
    }
}
"""

user_prompt_diagnose_code = """
{
    "user_prompt": {
        "task": "Extract only valid ICD-10 or DSM-5 diagnosis codes from the attached AUMI Behavioral Health document. The document may include handwritten text. Do not extract other types of codes.",
        "allowed_labels": [
            "ICD-10 Diagnoses",
            "ICD-10 Code",
            "Dx Code",
            "Diagnosis Code",
            "Primary Diagnosis Code",
            "Principal Diagnosis Code",
            "ICD Code",
            "DSM-5 Diagnoses",
            "Behavioral Health Diagnoses"
        ],
        "prohibited_codes": [
            "Procedure or Revenue Codes",
            "CPT, HCPCS codes",
            "Medications or Treatment codes"
        ],
        "output_format": {
            "json_structure": [
                {
                    "key": "diagnosis_code",
                    "value": "<string>",
                    "boundingBox": {
                        "topLeftX": "<integer>",
                        "topLeftY": "<integer>",
                        "bottomRightX": "<integer>",
                        "bottomRightY": "<integer>"
                    }
                }
            ],
            "empty_field_format": {
                "key": "diagnosis_code",
                "value": "",
                "boundingBox": {}
            }
        },
        "special_instructions": "Handwritten codes must be interpreted via OCR and their exact spelling and format preserved."
    }
}
"""

system_prompt_service_code = """
{
    "system_prompt": {
        "objective": "Extract only valid CPT or HCPCS procedure codes from medical or billing documents. Extraction is conditional on specific criteria, including selected checkboxes, unit values, or explicit headers. The output must be a precise JSON array.",
        "eligibility_rules": [
            "A row is eligible if its checkbox is selected.",
            "A row is eligible if its 'Units' or 'Quantity' value is greater than zero.",
            "A row is eligible if the page lacks checkboxes or units and the row is under a 'Procedure code(s)' heading.",
            "A row is eligible if a valid CPT/HCPCS code is found directly within labels like 'Requested Service:' or 'Service Requested:'."
        ],
        "target_fields": [
            "CPT\u00ae codes:",
            "CPT code",
            "HCPCS code",
            "HCPCS billing code",
            "Code / modifier",
            "Service code",
            "Requested Service"
        ],
        "extraction_steps": [
            "1. Analyze page layout to identify rows and associated elements (checkbox, code, units).",
            "2. Determine if a row is eligible based on the defined rules.",
            "3. Extract all valid CPT (5 digits) or HCPCS (1 letter + 4 digits) codes from eligible contexts.",
            "4. Preserve the exact text, including modifiers, and join multiple codes with a comma and space.",
            "5. Capture the bounding box for all characters of the extracted code string.",
            "6. Format the output as a JSON array."
        ],
        "output_format": {
            "valid_code_format": {
                "key": "service_code",
                "value": "<Code or comma-separated Codes>",
                "boundingBox": {
                    "topLeftX": "<integer>",
                    "topLeftY": "<integer>",
                    "bottomRightX": "<integer>",
                    "bottomRightY": "<integer>"
                }
            },
            "no_codes_found_format": [
                {
                    "key": "service_code",
                    "value": "",
                    "boundingBox": {}
                }
            ]
        }
    }
}
"""

user_prompt_service_code = """
{
    "user_prompt": {
        "task": "Extract procedure codes from the document.",
        "rules": [
            "A row is valid if its check-box is selected OR its units value > 0.",
            "A row is also valid if the page has no check-boxes or units and the row is under a 'Procedure code(s)' heading.",
            "A row is also valid if a valid CPT/HCPCS code explicitly appears inside labels such as 'Requested Service:', 'Requested Services:', or 'Service Requested:'.",
            "Return each eligible row as a JSON object with 'key', 'value', and 'boundingBox'."
        ],
        "output_format": {
            "single_object": {
                "key": "service_code",
                "value": "<Code or comma-separated Codes>",
                "boundingBox": {
                    "topLeftX": "<xmin>",
                    "topLeftY": "<ymin>",
                    "bottomRightX": "<xmax>",
                    "bottomRightY": "<ymax>"
                }
            },
            "no_codes_found": [
                {
                    "key": "service_code",
                    "value": "",
                    "boundingBox": {}
                }
            ]
        },
        "output_only": "Output only the final JSON array, with no other text."
    }
}
"""

system_prompt_service_from = """
{
    "system_prompt": {
        "objective": "Extract explicitly labeled start-related dates from prior authorization forms. The model must locate valid date fields that exactly match an approved keyword list and extract the corresponding date and its bounding box.",
        "allowed_keywords": [
            "Admission date",
            "Admit Date",
            "Start Date",
            "From Date",
            "Requested start Date",
            "Authorization from",
            "Date of service",
            "Date of service from",
            "DoS",
            "Date of Admission",
            "DOA",
            "DOS from",
            "Effective Date",
            "Start Date OR Admission Date",
            "Date of Diagnostic interview"
        ],
        "extraction_rules": [
            "Extract the date only if the label exactly matches a keyword from the approved list.",
            "Do not infer or guess labels or dates.",
            "Ignore unrelated timestamps.",
            "Return the bounding box for the date itself, not the label.",
            "Return only the first valid match found, based on a top-down, left-to-right reading order.",
            "If the label or date is missing, invalid, or unreadable, return an empty result."
        ],
        "output_format": {
            "valid_match": {
                "key": "service_from_date",
                "value": "<Extracted Date>",
                "boundingBox": {
                    "topLeftX": "<xmin>",
                    "topLeftY": "<ymin>",
                    "bottomRightX": "<xmax>",
                    "bottomRightY": "<ymax>"
                }
            },
            "no_match": {
                "key": "service_from_date",
                "value": "",
                "boundingBox": {}
            }
        }
    }
}
"""

user_prompt_service_from = """
{
    "user_prompt": {
        "task": "Extract the start-related date from the document using a provided keyword list.",
        "step_by_step_reasoning": [
            "1. Scan the entire document, including both printed and handwritten sections.",
            "2. Look for an exact match from the provided keyword list.",
            "3. If a match is found, extract the associated date value and its bounding box coordinates.",
            "4. Return only the first valid result found in reading order (top-down, left-to-right).",
            "5. If no valid keyword or date is found, return an empty result."
        ],
        "allowed_keywords": [
            "Admission date",
            "Admit Date",
            "Start Date",
            "From Date",
            "Requested start Date",
            "Authorization from",
            "Date of service",
            "Date of service from",
            "DoS",
            "Date of Admission",
            "DOA",
            "DOS from",
            "Effective Date",
            "Start Date OR Admission Date",
            "Date of Diagnostic interview"
        ],
        "output_format": {
            "valid_match": {
                "key": "service_from_date",
                "value": "<Extracted Date>",
                "boundingBox": {
                    "topLeftX": "<xmin>",
                    "topLeftY": "<ymin>",
                    "bottomRightX": "<xmax>",
                    "bottomRightY": "<ymax>"
                }
            },
            "no_match": {
                "key": "service_from_date",
                "value": "",
                "boundingBox": {}
            }
        }
    }
}
"""

system_prompt_adt_auth_prop = """
{
    "system_prompt": {
        "objective": "Strictly extract terms, codes, or facility names from a document image that exactly match a provided keyword list. Ignore anything not on the list. The output must be a single JSON object in a specific format.",
        "allowed_keywords_list": [
            "ABA", "Acute Detox", "Acute Detoxification", "Acute Psych", "Acute psychiatric inpatient",
            "Acute Rehab", "Acute RTC", "Acute SA Rehab", "adult life skills", "Adult Mental Health Rehab",
            "AMHR", "Applied Behavioral Analysis", "ASAM 2.1", "ASAM 2.5", "ASAM 2.7", "ASAM 3.1",
            "ASAM 3.3", "ASAM 3.5", "ASAM 3.7", "ASAM 4.0", "ASAM 4.0 or Any", "ASD",
            "Atlanticare Regional Medical Center", "Aurora", "Autism Spectrum Disorder", "Behavior Therapy",
            "Bellin, Catalpa Health", "Catalpa Health", "Center for Discovery", "Children's Crisis Residence",
            "Christ Hospital", "Christian Family Solutions", "Core Treatment Services", "CRC",
            "Crisis Residence", "crisis Risk assessment and intervention", "Crisis Stab",
            "Crisis Stabilization", "Crisis Stabilization,", "CSM", "CSU", "Day Treatment",
            "Day Tx", "Dominion Hospital Reflections", "Essentia Health Duluth",
            "facility based crisis", "facility based crisis S9484", "Family Services",
            "Gunderson Lutheran", "IHS", "In-Home Services", "Inpatient Detox", "Inpatient Psych",
            "Inpatient psychiatric", "Inpatient psychiatric rehab",
            "Inpatient Substance Use Disorder Treatment", "Intensive Crisis Residence", "IOP",
            "IOP mental health", "IOP substance abuse", "IOP substance use disorder", "IP Detox",
            "IP Psych", "Iris", "LOCADTR", "LOCADTR or Any", "Mayo Clinic", "MCHS Eau Clair",
            "Meriter", "mobile crisis", "Monte Nido", "New Bridge Medical Center", "Northwest Journey",
            "op resi", "Out of Network Override request Form", "outpatient residential treatment",
            "Partial Hospitalization Substance Abuse", "Partial Hospitilization Psych", "Pathways",
            "PHP", "PHP Psych", "PHP Substance Abuse", "PIC", "Pine Counseling", "PMIC", "PRTF",
            "PSR", "Psychiatric Intensive Care", "Psychiatric Medical Institutions for Children",
            "psychiatric residential treatment facilities", "Psychiatric residential treatment facility",
            "Psychiatric RTC", "psychosocial rehab", "psychosocial rehabilitation", "QRTP",
            "qualified residential treatment program", "rehab day", "Residential",
            "Residential Crisis Support", "Residential/Inpatient Substance Use Disorder Treatment",
            "Rogers", "RTC", "RTC Psych", "RTC SA", "RTC SUD", "St. Agnes",
            "St. Clare's Hospital", "St. Elizabeth", "St. Mary Ozaukee",
            "substance abuse residential treatment", "Substance use disorder services", "SUD",
            "SUD RTC", "Tarrant County (MHMR)", "Tarrant County (MHMR) or Any", "Telecare",
            "Tellurian", "TGH", "ThedaCare", "Therapeutic Group Home",
            "therapeutic repetitive transcranial magnetic stimulation", "TLS Behavioral Crisis Resource Center",
            "TMS", "Transcranial Magnetic Stimulation", "Veritas", "Willow Creek",
            "Wood County Human Services", "URGENT", "urgent"
        ],
        "output_format": {
            "json_structure": [
                {
                    "key": "additional_auth_properties",
                    "value": "<Comma-separated list of all exact keywords found>",
                    "boundingBox": {
                        "topLeftX": "<xmin>",
                        "topLeftY": "<ymin>",
                        "bottomRightX": "<xmax>",
                        "bottomRightY": "<ymax>"
                    }
                }
            ],
            "no_match_format": {
                "key": "additional_auth_properties",
                "value": "",
                "boundingBox": {
                    "topLeftX": 0,
                    "topLeftY": 0,
                    "bottomRightX": 0,
                    "bottomRightY": 0
                }
            }
        },
        "step-by-step_reasoning": [
            "1. Scan the entire document image.",
            "2. Compare each piece of text with the allowed keyword list for an exact match.",
            "3. Collect all exact matches.",
            "4. Combine all matched keywords into a single comma-separated string for the 'value' field.",
            "5. Report the tight bounding box enclosing all found keywords.",
            "6. Output the final JSON array strictly as instructed."
        ]
    }
}
"""

user_prompt_adt_auth_prop = """
{
    "user_prompt": {
        "task": "Review the attached document image and extract only the terms, codes, or facility names that are an exact match to the provided list. Ignore all other text.",
        "allowed_keywords_list": [
            "ABA", "Acute Detox", "Acute Detoxification", "Acute Psych", "Acute psychiatric inpatient",
            "Acute Rehab", "Acute RTC", "Acute SA Rehab", "adult life skills", "Adult Mental Health Rehab",
            "AMHR", "Applied Behavioral Analysis", "ASAM 2.1", "ASAM 2.5", "ASAM 2.7", "ASAM 3.1",
            "ASAM 3.3", "ASAM 3.5", "ASAM 3.7", "ASAM 4.0", "ASAM 4.0 or Any", "ASD",
            "Atlanticare Regional Medical Center", "Aurora", "Autism Spectrum Disorder", "Behavior Therapy",
            "Bellin, Catalpa Health", "Catalpa Health", "Center for Discovery", "Children's Crisis Residence",
            "Christ Hospital", "Christian Family Solutions", "Core Treatment Services", "CRC",
            "Crisis Residence", "crisis Risk assessment and intervention", "Crisis Stab",
            "Crisis Stabilization", "Crisis Stabilization,", "CSM", "CSU", "Day Treatment",
            "Day Tx", "Dominion Hospital Reflections", "Essentia Health Duluth",
            "facility based crisis", "facility based crisis S9484", "Family Services",
            "Gunderson Lutheran", "IHS", "In-Home Services", "Inpatient Detox", "Inpatient Psych",
            "Inpatient psychiatric", "Inpatient psychiatric rehab",
            "Inpatient Substance Use Disorder Treatment", "Intensive Crisis Residence", "IOP",
            "IOP mental health", "IOP substance abuse", "IOP substance use disorder", "IP Detox",
            "IP Psych", "Iris", "LOCADTR", "LOCADTR or Any", "Mayo Clinic", "MCHS Eau Clair",
            "Meriter", "mobile crisis", "Monte Nido", "New Bridge Medical Center", "Northwest Journey",
            "op resi", "Out of Network Override request Form", "outpatient residential treatment",
            "Partial Hospitalization Substance Abuse", "Partial Hospitilization Psych", "Pathways",
            "PHP", "PHP Psych", "PHP Substance Abuse", "PIC", "Pine Counseling", "PMIC", "PRTF",
            "PSR", "Psychiatric Intensive Care", "Psychiatric Medical Institutions for Children",
            "psychiatric residential treatment facilities", "Psychiatric residential treatment facility",
            "Psychiatric RTC", "psychosocial rehab", "psychosocial rehabilitation", "QRTP",
            "qualified residential treatment program", "rehab day", "Residential",
            "Residential Crisis Support", "Residential/Inpatient Substance Use Disorder Treatment",
            "Rogers", "RTC", "RTC Psych", "RTC SA", "RTC SUD", "St. Agnes",
            "St. Clare's Hospital", "St. Elizabeth", "St. Mary Ozaukee",
            "substance abuse residential treatment", "Substance use disorder services", "SUD",
            "SUD RTC", "Tarrant County (MHMR)", "Tarrant County (MHMR) or Any", "Telecare",
            "Tellurian", "TGH", "ThedaCare", "Therapeutic Group Home",
            "therapeutic repetitive transcranial magnetic stimulation", "TLS Behavioral Crisis Resource Center",
            "TMS", "Transcranial Magnetic Stimulation", "Veritas", "Willow Creek",
            "Wood County Human Services", "URGENT", "urgent"
        ],
        "extraction_rules": "Match each piece of text with the list. Collect exact matches only (no synonyms, no close matches).",
        "output_format": {
            "json_structure": [
                {
                    "key": "additional_auth_properties",
                    "value": "<Comma-separated list of all exact keywords found>",
                    "boundingBox": {
                        "topLeftX": "<xmin>",
                        "topLeftY": "<ymin>",
                        "bottomRightX": "<xmax>",
                        "bottomRightY": "<ymax>"
                    }
                }
            ]
        },
        "no_match_format": {
            "value": ""
        }
    }
}
"""

system_prompt_level_care = """
{
    "system_prompt": {
        "objective": "Identify and extract 'Level of Care' (LOC) values from healthcare and administrative forms. The model must handle all formats, including checkboxes, text fields, and tables, and return the findings in a strict JSON array format.",
        "level_of_care_definition": "The intensity or type of medical services a patient requires.",
        "allowed_labels": [
            "Level of Care",
            "LOC"
        ],
        "extraction_rules": [
            "Extract values from checkboxes that are marked with a filled marker (tick, check, cross).",
            "Extract the value that follows a 'Level of Care' or 'LOC' text field label.",
            "In tables, extract row-wise data from the 'Level of Care' column only if corresponding data cells are filled.",
            "If multiple values are found, they should be combined into a comma-separated string.",
            "Bounding box coordinates must be provided for the extracted value(s).",
            "If no values are found, the 'value' field should be an empty string and the 'boundingBox' should be an empty object."
        ],
        "output_format": {
            "json_structure": [
                {
                    "key": "level_of_care",
                    "value": "<extracted_value(s)>",
                    "boundingBox": {
                        "topLeftX": "<xmin>",
                        "topLeftY": "<ymin>",
                        "bottomRightX": "<xmax>",
                        "bottomRightY": "<ymax>"
                    }
                }
            ],
            "multiple_values_format": "Combine multiple values into a single comma-separated string.",
            "no_value_found_format": {
                "key": "level_of_care",
                "value": "",
                "boundingBox": {}
            }
        }
    }
}
"""

user_prompt_level_care = """
{
    "user_prompt": {
        "task": "Extract all 'Level of Care' values from the provided document images, checking all formats (checkbox, text, and table).",
        "reasoning_chain": [
            "Step 1: Locate sections with the phrase 'Level of Care' or 'LOC'.",
            "Step 2: Classify the format as checkbox, text field, or tabular.",
            "Step 3: Extract based on the format:",
            " - For checkboxes: identify filled markers and extract the corresponding labels.",
            " - For text fields: extract the value following the 'Level of Care' label.",
            " - For tables: extract entries from the 'Level of Care' column where data cells are filled.",
            "Step 4: Standardize the output into a single JSON object in an array."
        ],
        "output_format": {
            "values_found": {
                "key": "level_of_care",
                "value": "<extracted_value(s)>",
                "boundingBox": {
                    "topLeftX": "<xmin>",
                    "topLeftY": "<ymin>",
                    "bottomRightX": "<xmax>",
                    "bottomRightY": "<ymax>"
                }
            },
            "multiple_values_found": {
                "key": "level_of_care",
                "value": "<value1>, <value2>",
                "boundingBox": {
                    "topLeftX": "<xmin>",
                    "topLeftY": "<ymin>",
                    "bottomRightX": "<xmax>",
                    "bottomRightY": "<ymax>"
                }
            },
            "no_value_found": {
                "key": "level_of_care",
                "value": "",
                "boundingBox": {}
            }
        }
    }
}
"""

system_prompt_servicing = """
{
    "system_prompt": {
        "objective": "Identify and extract all servicing provider and facility details from healthcare forms, handling text fields, forms, and tables. The output must be a strict JSON array with keys for various fields like names, addresses, NPIs, and contact info.",
        "servicing_details_definition": "Information about healthcare providers and facilities that render services, including names, addresses, NPIs, specialties, TINs, and contact information.",
        "extraction_rules": [
            "Analyze text fields, form fields, and tabular data for servicing details.",
            "Extract only clearly marked or filled values.",
            "Use a multi-zoom OCR pipeline for handwritten or low-confidence text.",
            "Separate first name, last name, and full name for both providers and facilities.",
            "Extract address line 1 and line 2 separately.",
            "Exclude member-related or referring provider data.",
            "Output results in a strict JSON array format with 'key', 'value', and 'boundingBox'."
        ],
        "output_format": {
            "json_structure": [
                {
                    "key": "<field_name>",
                    "value": "<extracted_value>",
                    "boundingBox": {
                        "topLeftX": "<xmin>",
                        "topLeftY": "<ymin>",
                        "bottomRightX": "<xmax>",
                        "bottomRightY": "<ymax>"
                    }
                }
            ],
            "field_names": [
                "servicing_provider_address_line1", "servicing_provider_address_line2",
                "servicing_provider_first_name", "servicing_provider_last_name",
                "servicing_provider_full_name", "servicing_provider_npi",
                "servicing_provider_city", "servicing_provider_state",
                "servicing_provider_zipcode", "servicing_provider_tin",
                "servicing_provider_specialty", "servicing_facility_first_name",
                "servicing_facility_last_name", "servicing_facility_full_name",
                "servicing_facility_address_line1", "servicing_facility_address_line2",
                "servicing_facility_npi", "servicing_facility_tin",
                "servicing_facility_city", "servicing_facility_state",
                "servicing_facility_zipcode", "servicing_facility_specialty"
            ]
        }
    }
}
"""

user_prompt_servicing = """
{
    "user_prompt": {
        "task": "Extract all Servicing Details from the document, including provider and facility information such as names, addresses, NPIs, and contact details.",
        "reasoning_chain": [
            "Step 1: Locate sections with labels like 'Servicing Provider', 'Servicing Facility', 'Rendering Provider', or 'Attending Provider'.",
            "Step 2: Classify the information format (text field, form field, or table).",
            "Step 3: Extract based on the format, handling names, addresses, and tabular data.",
            "Step 4: Apply special processing for multi-zoom OCR, name separation, address separation, and contact information.",
            "Step 5: Return a standardized JSON output."
        ],
        "output_format": {
            "valid_values_found": [
                {
                    "key": "servicing_provider_address_line1",
                    "value": "<extracted_value>",
                    "boundingBox": {
                        "topLeftX": "<xmin>",
                        "topLeftY": "<ymin>",
                        "bottomRightX": "<xmax>",
                        "bottomRightY": "<ymax>"
                    }
                },
                "<... and other fields as per the list ...>"
            ],
            "no_values_found": [
                {
                    "key": "<field_name>",
                    "value": "",
                    "boundingBox": {}
                },
                "<... for all fields as per the list ...>"
            ]
        }
    }
}
"""

user_prompt_basic_test = """
Extract the patient's name from this document.
"""

system_prompt_basic_test = """
You are a helpful assistant.
"""

PROMPT_DICTIONARY = {
        "servicing": (user_prompt_servicing, system_prompt_servicing),
        "level_care": (user_prompt_level_care, system_prompt_level_care),
        "adt_auth_prop":(user_prompt_adt_auth_prop, system_prompt_adt_auth_prop),
        "service_from": (user_prompt_service_from, system_prompt_service_from),
        "service_code":(user_prompt_service_code, system_prompt_service_code),
        "diagnose_code":(user_prompt_diagnose_code, system_prompt_diagnose_code),
        "member":(user_prompt_member, system_prompt_member),
        "validate":(user_prompt_basic_test, system_prompt_basic_test)
    }

user_prompt = user_prompt_level_care
system_prompt = system_prompt_level_care

## chose from _memeber, _service_code, _diagnose_code, _service_from, _adt_auth_prop, _level_care, _servicing

# user_prompt = user_prompt_servicing
# system_prompt = system_prompt_servicing

# Get the first model served by vLLM
model = client.models.list().data[0].id


# Helper to encode image as base64
def encode_base64_content_from_localimg(image_path: str) -> str:
    with open(image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read())
    return encoded_image.decode("utf-8")


# Inference using single image
def run_single_image_qwen(image_file: str) -> None:
    print(f"Running Qwen inference for: {image_file}")

    try:
        image_base64 = encode_base64_content_from_localimg(image_file)
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                },
            ],
            model=model,
            max_tokens=1024,
            temperature=0.1,
            top_p=0.001,
            extra_body={
                "top_k": 1,
                "repetition_penalty": 1.05,
            },
        )
        print(
            f"[{os.path.basename(image_file)}] Response:",
            response.choices[0].message.content,
        )
        try:
            os.makedirs(output_folder, exist_ok=True)
            print(
                f"Directory '{output_folder}' created successfully or already exists."
            )
        except OSError as e:
            print(f"Error creating directory '{output_folder}': {e}")

        with open(
            output_folder + os.sep + os.path.basename(image_file.replace(" ", "_")) + ".txt", "w"
        ) as f:
            f.write(response.choices[0].message.content)
        # with open(os.path.basename(image_file) + '.txt', 'w') as f:
        #     f.write(response.choices[0].message.content)
    except Exception as e:
        print(f"[{os.path.basename(image_file)}] Failed with error: {e}")


# Process all images in a folder using multiprocessing
def run_batch_inference(
    folder_path: str, prompts: str, output: str, num_processes: int = None, num_images: int = None,
) -> None:

    global user_prompt
    global system_prompt
    global output_folder


    user_prompt, system_prompt = PROMPT_DICTIONARY.get(prompts, ("", ""))
    output_folder = output


    image_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]  # [:2]
    if not image_files:
        print("No image files found in folder.")
        return

    if num_images is not None and num_images < len(image_files):
        image_files = image_files[0:num_images]

    print(f"Running inference on {len(image_files)} images...")

    if num_processes is None:
        num_processes = num_processes or min(
            multiprocessing.cpu_count(), 32
        )  # len(image_files))
        num_processes = len(image_files)

    t1 = time.time()
    _pool = multiprocessing.Pool(processes=1)
    _pool.apply_async(run_single_image_qwen, args=(image_files,))
    time.sleep(5)
    t2 = time.time()
    with multiprocessing.Pool(num_processes) as pool:
        pool.map(partial(run_single_image_qwen), image_files)
    t3 = time.time()
    _pool.close()
    _pool.join()
    t4 = time.time()

    print(f"\n-----Stats: ----- ")
    print(f"1-core query time (excluded) [s]: {t2-t1+t4-t3}")
    print(f"All-core query time (performance) [s]: {t3-t2}")

    with open(os.path.join(output_folder, "performance_results.txt"), "w") as file:
        file.write(f"1-core query time (excluded) [s]: {t2-t1+t4-t3}")
        file.write(f"All-core query time (performance) [s]: {t3-t2}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--folder",
        default="/data/52-pages/",
        help="the path to the dataset folder",
    )
    parser.add_argument(
        "-np",
        "--num_processor",
        default=multiprocessing.cpu_count(),
        type=int,
        help="the number of processors",
    )
    parser.add_argument(
        "-ni",
        "--num_images",
        type=int,
        help="number of images to run (will be all if not specified)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="./output",
        help="Location to write output data"
    )
    parser.add_argument(
        "-p",
        "--prompt",
        default="diagnose_code",
        choices=["servicing", "level_care", "adt_auth_prop", "service_from", "service_code", "diagnose_code", "member", "validate"],
        help="prompts to use"
    )
    args = parser.parse_args()


    # folder_path = args.folder
    # folder_path = "/devops/sgohari/tests/jira/hs-6456/50-pages/50-pages"  # Change this to your image folder
    run_batch_inference(
        folder_path=args.folder,
        num_processes=args.num_processor,
        num_images=args.num_images,
        prompts=args.prompt,
        output=args.output
    )
