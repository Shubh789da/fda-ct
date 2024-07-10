import requests
import pandas as pd
from dateutil.parser import parse
from dateutil.parser import ParserError  

def get_clinical_trials_data(COND):
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    params = {
        "query.term": str(COND),
        "pageSize": 1000,
        "pageToken": None  # Set initial page token to None
    }

    all_studies = {}
    i = 0
    while True:
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            studies = data.get("studies", [])
            
            if i == 0:
                all_studies.update(data)  # Add the studies from this page to the dictionary
                page_token = data.get("nextPageToken")
            elif i > 0:
                # Extend the studies list with new studies
                all_studies["studies"].extend(data.get("studies", []))
                page_token = data.get("nextPageToken")
            if not page_token:
                break  # Exit the loop when there are no more pages
            params['pageToken'] = page_token  # Set the page token for the next request
            i += 1
            #print(f"Page {i} processed")
        else:
            print(f"Error fetching data: {response.status_code}")
            break  # Exit on error

    def normalize_study(study):
        flat_data = {}
        
        # Extract identification module
        identification = study.get('protocolSection', {}).get('identificationModule', {})
        flat_data['nctId'] = identification.get('nctId')
        flat_data['organization'] = identification.get('organization', {}).get('fullName')
        flat_data['organizationType'] = identification.get('organization', {}).get('class')
        flat_data['briefTitle'] = identification.get('briefTitle')
        flat_data['officialTitle'] = identification.get('officialTitle')
        
        # Extract status module
        status = study.get('protocolSection', {}).get('statusModule', {})
        flat_data['statusVerifiedDate'] = status.get('statusVerifiedDate')
        flat_data['overallStatus'] = status.get('overallStatus')
        flat_data['hasExpandedAccess'] = status.get('expandedAccessInfo', {}).get('hasExpandedAccess')
        flat_data['startDate'] = status.get('startDateStruct', {}).get('date')
        flat_data['completionDate'] = status.get('completionDateStruct', {}).get('date')
        flat_data['completionDateType'] = status.get('completionDateStruct', {}).get('type')
        flat_data['studyFirstSubmitDate'] = status.get('studyFirstSubmitDate')
        flat_data['studyFirstPostDate'] = status.get('studyFirstPostDateStruct', {}).get('date')
        flat_data['lastUpdatePostDate'] = status.get('lastUpdatePostDateStruct', {}).get('date')
        flat_data['lastUpdatePostDateType'] = status.get('lastUpdatePostDateStruct', {}).get('type')

        #Results status
        flat_data['HasResults'] = study.get('hasResults')
        
        # Extract sponsor collaborators module
        sponsor = study.get('protocolSection', {}).get('sponsorCollaboratorsModule', {})
        flat_data['responsibleParty'] = sponsor.get('responsibleParty', {}).get('oldNameTitle')
        flat_data['leadSponsor'] = sponsor.get('leadSponsor', {}).get('name')
        flat_data['leadSponsorType'] = sponsor.get('leadSponsor', {}).get('class')
        flat_data['collaborators'] = ', '.join([collab.get('name') for collab in sponsor.get('collaborators', [])])
        flat_data['collaboratorsType'] = ', '.join([collab.get('class') for collab in sponsor.get('collaborators', [])])
        
        # Extract description module
        description = study.get('protocolSection', {}).get('descriptionModule', {})
        flat_data['briefSummary'] = description.get('briefSummary')
        flat_data['detailedDescription'] = description.get('detailedDescription')
        
        # Extract conditions module
        conditions = study.get('protocolSection', {}).get('conditionsModule', {})
        flat_data['conditions'] = ', '.join(conditions.get('conditions', []))
        
        # Extract design module
        design = study.get('protocolSection', {}).get('designModule', {})
        flat_data['studyType'] = design.get('studyType')
        flat_data['phases'] = ', '.join(design.get('phases', []))
        flat_data['allocation'] = design.get('designInfo', {}).get('allocation')
        flat_data['interventionModel'] = design.get('designInfo', {}).get('interventionModel')
        flat_data['primaryPurpose'] = design.get('designInfo', {}).get('primaryPurpose')
        flat_data['masking'] = design.get('designInfo', {}).get('maskingInfo', {}).get('masking')
        flat_data['whoMasked'] = ', '.join(design.get('designInfo', {}).get('maskingInfo', {}).get('whoMasked', []))
        flat_data['enrollmentCount'] = design.get('enrollmentInfo', {}).get('count')
        flat_data['enrollmentType'] = design.get('enrollmentInfo', {}).get('type')
        
        # Extract arms interventions module
        arms = study.get('protocolSection', {}).get('armsInterventionsModule', {}).get('armGroups', [])
        flat_data['arms'] = ', '.join([arm.get('label') for arm in arms])
        flat_data['interventions'] = ', '.join([', '.join(arm.get('interventionNames', [])) for arm in arms])

        # Extract the outcome module
        outcome = study.get('protocolSection', {}).get('outcomesModule', {})
        flat_data['primaryOutcomes'] = '\n'.join([
                                                    f"Primary Outcome {i + 1}: {primary_outcome.get('measure', None) or 'None'}"
                                                    for i, primary_outcome in enumerate(outcome.get('primaryOutcomes', []))
                                                ])
        flat_data['secondaryOutcomes'] = '\n'.join([
                                                    f"Secondary Outcome {i + 1}: {primary_outcome.get('measure', None) or 'None'}"
                                                    for i, primary_outcome in enumerate(outcome.get('secondaryOutcomes', []))
                                                ])
        #Extract Eligibility
        eligibility = study.get('protocolSection',{}).get('eligibilityModule',{})
        flat_data['eligibilityCriteria'] = eligibility.get('eligibilityCriteria')
        flat_data['healthyVolunteers'] = eligibility.get('healthyVolunteers')
        flat_data['eligibilityGender'] = eligibility.get('sex')
        flat_data['eligibilityMinimumAge'] = eligibility.get('minimumAge')
        flat_data['eligibilityMaximumAge'] = eligibility.get('maximumAge')
        flat_data['eligibilityStandardAges'] = eligibility.get('stdAges')

        #Extract the locations
        locations = study.get('protocolSection',{}).get('contactsLocationsModule',{}).get('locations',{})
        if locations is not None:
            flat_data['LocationName'] = ', '.join(set(location.get('facility') or '' for location in locations)) if locations is not None else ''
            flat_data['city'] = ', '.join(set(location.get('city') or '' for location in locations)) if locations is not None else '' 
            flat_data['state'] = ', '.join(set(location.get('state') or '' for location in locations)) if locations is not None else '' 
            flat_data['country'] = ', '.join(set(location.get('country') or '' for location in locations)) if locations is not None else '' 

        return flat_data

    # Normalize all studies
    normalized_data = [normalize_study(study) for study in all_studies.get('studies', [])]

    # Convert to DataFrame
    df = pd.DataFrame(normalized_data)

    def parse_date(date_str):
        if pd.isna(date_str):
            return pd.NaT
        if isinstance(date_str, list):
            date_str = date_str[0] if date_str else None
        if not date_str:
            return pd.NaT
        try:
            # Parse the date, set day to 1 if only year and month are provided
            parsed_date = parse(date_str, default=parse('2000-01-01'))
            if len(date_str) <= 7:  # If only year or year-month is provided
                return parsed_date.replace(day=1)
            return parsed_date
        except ParserError:
            return pd.NaT

    # Convert date columns
    date_columns = ['startDate', 'completionDate', 'studyFirstSubmitDate', 'studyFirstPostDate', 'lastUpdatePostDate']
    
    for col in date_columns:
        df[col] = df[col].apply(parse_date)
    
    return df
    

# Example usage:
# df = get_clinical_trials_data("Novartis")
