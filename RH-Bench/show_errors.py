import json

f = '/iopsstor/scratch/cscs/raghavthind/rh-bench/results/judged_responses_gemma4.json'
results = json.load(open(f))

SEP = '='*70

for subset in ['halu', 'reason']:
    # Find incorrect answers with good thinking (50-600 words) and meaningful response
    wrong = [r for r in results
             if r['subset'] == subset
             and not r.get('is_correct', True)
             and 50 < r.get('thinking_words', 0) < 600
             and len(r.get('clean_response', '').strip()) > 80
             and r.get('has_thinking')]

    if not wrong:
        print('No wrong examples found for %s with these filters' % subset)
        continue

    # Pick one near the middle thinking length
    wrong.sort(key=lambda r: r['thinking_words'])
    pick = wrong[len(wrong)//2]

    raw = pick.get('raw_response', '')
    if '<think>' in raw and '</think>' in raw:
        thinking = raw.split('<think>')[1].split('</think>')[0].strip()
    else:
        thinking = '(no <think> tags)'

    print('\n' + SEP)
    print('SUBSET:   %s  [INCORRECT]' % pick['subset'])
    print('ID:       %s' % pick['id'])
    print('Q-TYPE:   %s' % pick['question_type'])
    print('IMAGE:    /capstor/store/cscs/swissai/a0174/benchmarks/RH-Bench/%s' % pick['image'])
    print('ELAPSED:  %.1fs  |  THINKING: %d words' % (pick['elapsed_s'], pick['thinking_words']))
    print(SEP)
    print('\nQUESTION:\n%s' % pick['question'])
    print('\nGROUND TRUTH:\n%s' % pick['gt_answer'])
    print('\n--- THINKING ---\n%s' % thinking)
    print('\n--- FINAL ANSWER ---\n%s' % pick['clean_response'])
    print('\n--- JUDGE ---')
    print('Correct: %s' % pick.get('is_correct'))
    print('Reason:  %s' % pick.get('evaluation_reason', '')[:300])
    if pick.get('hallucination_score') is not None:
        print('Hallucination score: %d/5' % pick['hallucination_score'])
    if pick.get('pred_letter'):
        print('Predicted: %s  |  GT letter: %s' % (pick.get('pred_letter'), pick.get('gt_letter')))
    print(SEP)
