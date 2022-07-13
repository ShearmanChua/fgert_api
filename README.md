# fget_api

api data format:

dataframe with doc_id, text, text_tokens and entities

Example:

doc_id: 5603098-4

Text: Alyan Muhammad Ali al - Wa'eli ( Arabic : ) ( born in 1970 in Yemen ) became wanted in 2002 , by the United States Department of Justice 's FBI , which was then seeking information about his identity and whereabouts .

text_tokens: ['Alyan', 'Muhammad', 'Ali', 'al', '-', ""Wa'eli"", '(', 'Arabic', ':', ')', '(', 'born', 'in', '1970', 'in', 'Yemen', ')', 'became', 'wanted', 'in', '2002', ',', 'by', 'the', 'United', 'States', 'Department', 'of', 'Justice', ""'s"", 'FBI', ',', 'which', 'was', 'then', 'seeking', 'information', 'about', 'his', 'identity', 'and', 'whereabouts', '.']

entities: [{'labels': ['/person'], 'start': 0, 'end': 6, 'mention': ""Alyan Muhammad Ali al - Wa'eli"", 'mention_id': '5603098-0'}, {'labels': ['/language'], 'start': 7, 'end': 8, 'mention': 'Arabic', 'mention_id': '5603098-1'}, {'labels': ['/location', '/location/country'], 'start': 15, 'end': 16, 'mention': 'Yemen', 'mention_id': '5603098-2'}, {'labels': ['/organization', '/organization/government_agency'], 'start': 24, 'end': 29, 'mention': 'United States Department of Justice', 'mention_id': '5603098-3'}, {'labels': ['/organization', '/organization/government_agency'], 'start': 30, 'end': 31, 'mention': 'FBI', 'mention_id': '5603098-4'}]

* labels can be left as an empty string ''