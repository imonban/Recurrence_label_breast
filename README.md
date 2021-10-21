# Recurrence_label_breast
We developed an NLP method that enables identification of the occurrence and timing of metastatic breast cancer recurrence from EMRs. This approach may be adaptable to other cancer sites and could help to unlock the potential of EMRs for research on real-world cancer outcomes.

In order to run the labeling code, please following the following steps - 

1. Create a model and download the trained models there - https://drive.google.com/drive/folders/1vEp5SsW93oX1hMDJkIq2qhQrnUOkqAdL?usp=sharing
2. Create a outcome folder 
3. Run python Main.py with clinic notes saved in a excel file with the follwing fields 
          ANON_ID - Patient identified
          NOTE_TYPE - e.g. 'Discharge', 'Oncology consultatiions', 'ICU notes'
          NOTE_DATE -  Date of the encounter
          NOTE - String blob
4. Model will save the output in ./outcome folder
 
# Reference 
Banerjee, Imon, Selen Bozkurt, Jennifer Lee Caswell-Jin, Allison W. Kurian, and Daniel L. Rubin. "Natural language processing approaches to detect the timeline of metastatic recurrence of breast cancer." JCO clinical cancer informatics 3 (2019): 1-12.
