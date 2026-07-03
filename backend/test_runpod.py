"""
test_runpod.py
--------------
Standalone script to test the MedGemma RunPod endpoint directly.
Run from the backend directory:

    conda activate nemotron310
    python test_runpod.py
"""

from correction.medgemma import correct_transcript

SAMPLE_CONVERSATION = [
    {"speaker": "Doctor",  "text": "You came with chest pain since morning correct."},
    {"speaker": "Patient", "text": "Yes doctor heavy chest pain going to my left arm with sweating."},
    {"speaker": "Doctor",  "text": "Electrocardiograph shows ST segment elevation in leads two three and a V F suggesting inferior wall myocardial in fraction."},
    {"speaker": "Doctor",  "text": "Cardiac propane one is elevated at two point five nanograms per milliliter."},
    {"speaker": "Doctor",  "text": "We will load you with as prin three hundred milligrams and clopy dog rel three hundred milligrams."},
    {"speaker": "Doctor",  "text": "Start at over statin eighty milligrams at night and echo cardio graphy tomorrow morning."},
    {"speaker": "Patient", "text": "Doctor I am already taking sorbit rate five milligrams under the tongue when pain comes."},
    {"speaker": "Doctor",  "text": "Good we will also start met o pro lol twenty five milligrams twice daily if blood pressure allows."},
]

if __name__ == "__main__":
    result = correct_transcript(SAMPLE_CONVERSATION)

    print("\n" + "═" * 70)
    print("FINAL CORRECTED CONVERSATION")
    print("═" * 70)
    for turn in result["corrected_conversation"]:
        print(f"  [{turn['speaker']}] {turn['text']}")
    print("═" * 70)
    print(f"Job ID            : {result['job_id']}")
    print(f"Execution time    : {result['execution_time_ms']} ms")
    print(f"Turns changed     : {len(result['changes'])}/{len(SAMPLE_CONVERSATION)}")