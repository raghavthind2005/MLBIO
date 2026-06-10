import json

f = '/iopsstor/scratch/cscs/raghavthind/rh-bench/results/judged_responses_gemma4.json'
results = json.load(open(f))

# Pick 1 halu + 1 reason with good thinking (100-800 words, complete response)
examples = []
for subset in ['halu', 'reason']:
    candidates = [r for r in results
                  if r['subset'] == subset
                  and 100 < r.get('thinking_words', 0) < 800
                  and len(r.get('clean_response','').strip()) > 100
                  and r.get('has_thinking')]
    if candidates:
        # pick one with middling thinking length for readability
        candidates.sort(key=lambda r: r['thinking_words'])
        pick = candidates[len(candidates)//2]
        examples.append(pick)

SEP = '='*70

for rec in examples:
    raw = rec.get('raw_response', '')
    # Extract thinking from raw_response
    if '<think>' in raw and '</think>' in raw:
        thinking = raw.split('<think>')[1].split('</think>')[0].strip()
    else:
        thinking = '(no <think> tags found)'

    print('\n' + SEP)
    print('SUBSET:   %s' % rec['subset'])
    print('ID:       %s' % rec['id'])
    print('Q-TYPE:   %s' % rec['question_type'])
    print('IMAGE:    /capstor/store/cscs/swissai/a0174/benchmarks/RH-Bench/%s' % rec['image'])
    print('ELAPSED:  %.1fs  |  THINKING: %d words' % (rec['elapsed_s'], rec['thinking_words']))
    print(SEP)
    print('\nQUESTION:\n%s' % rec['question'])
    print('\nGROUND TRUTH:\n%s' % rec['gt_answer'])
    print('\n--- THINKING ---\n%s' % thinking)
    print('\n--- FINAL ANSWER ---\n%s' % rec['clean_response'])
    print('\n--- JUDGE ---')
    print('Correct: %s' % rec.get('is_correct'))
    print('Reason:  %s' % rec.get('evaluation_reason', '')[:200])
    if rec.get('hallucination_score') is not None:
        print('Hallucination score: %d/5' % rec['hallucination_score'])
    print(SEP)
