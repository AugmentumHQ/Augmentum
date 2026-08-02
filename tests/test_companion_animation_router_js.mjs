// tests/test_companion_animation_router_js.mjs
//
// Pure-Node tests for the companion animation adapter. The browser-owned
// systems stay mocked; this verifies the reusable routing contract.
//
// Run with:
//   node tests/test_companion_animation_router_js.mjs

import {
  CompanionAnimationRouter,
  normalizeCompanionAnimationIntent,
  normalizeCompanionPoseIntent,
  __test,
} from '../ui/scripts/companion-animation-router.js';

let _failed = 0;
let _ran = 0;

function assert(cond, label) {
  _ran++;
  if (cond) console.log(`PASS ${label}`);
  else {
    _failed++;
    console.error(`FAIL ${label}`);
  }
}

function assertEq(actual, expected, label) {
  _ran++;
  if (JSON.stringify(actual) === JSON.stringify(expected)) {
    console.log(`PASS ${label}`);
  } else {
    _failed++;
    console.error(`FAIL ${label}\n  expected: ${JSON.stringify(expected)}\n    actual: ${JSON.stringify(actual)}`);
  }
}

function makeConductor() {
  return {
    playCalls: [],
    playByIdCalls: [],
    setBiasCalls: [],
    setBias(value) { this.setBiasCalls.push(value); },
    play(intent, options = {}) {
      this.playCalls.push({ intent, options });
      return Promise.resolve({ id: 'picked' });
    },
    playById(id, options = {}) {
      this.playByIdCalls.push({ id, options });
      return Promise.resolve({ id });
    },
  };
}

function makeAvatarState() {
  return {
    _poseIntent: 'idle_standing',
    animator: {
      poseCalls: [],
      setPoseIntent(family) { this.poseCalls.push(family); },
    },
  };
}

(function dominantAudioKindPrefersSpeechThenMusic() {
  assertEq(__test._dominantAudioKind(['music', 'speech']), 'speech',
    'audio kind routing lets speech preempt music');
  assertEq(__test._dominantAudioKind(['sfx', 'ambient']), 'ambient',
    'audio kind routing ignores sfx and keeps ambient');
})();

(function normalizesAnimationIntent() {
  const intent = normalizeCompanionAnimationIntent({
    topic: 'behavior.animation_intent',
    payload: {
      roles: ['wave', '', 'greet'],
      mood: { warmth: 1.4, energy: 0.5, focus: 'bad' },
      priority: 'explicit',
    },
  });
  assertEq(intent.roles, ['wave', 'greet'], 'roles normalize to non-empty strings');
  assertEq(intent.emotion, { warmth: 1, energy: 0.5 }, 'emotion clamps numeric keys');
  assert(intent.explicit === true, 'explicit priority marks explicit dispatch');
})();

(function normalizesPoseVerbIntent() {
  const pose = normalizeCompanionPoseIntent({
    topic: 'behavior.animation_intent',
    payload: {
      pose_verb: 'surface-attention',
      pose_duration_ms: 99999,
    },
  });
  assertEq(pose.family, 'idle_lefthip', 'pose verb resolves to an existing family');
  assertEq(pose.verb, 'surface_attention', 'pose verb normalizes separators');
  assertEq(pose.durationMs, 45000, 'pose duration clamps to max');
})();

(function rejectsUnknownPoseFamily() {
  const pose = normalizeCompanionPoseIntent({
    payload: { pose_family: 'moonwalk_idle' },
  });
  assertEq(pose, null, 'unknown pose family is ignored');
})();

(function normalizedPoseFamilyCanBeAppliedAgain() {
  const pose = normalizeCompanionPoseIntent({
    payload: { pose_family: 'formal', pose_duration_ms: 1200 },
  });
  const applied = normalizeCompanionPoseIntent(pose);
  assertEq(applied.family, 'formal', 'normalized pose family survives a second normalization pass');
  assertEq(applied.durationMs, 1200, 'normalized pose duration survives a second normalization pass');
})();

(function restAndPresenceVerbsUsePromotedFamilies() {
  const settled = normalizeCompanionPoseIntent({ payload: { pose_verb: 'settled' } });
  const confident = normalizeCompanionPoseIntent({ payload: { pose_verb: 'confident' } });
  assertEq(settled.family, 'idle_holding', 'settled verb reuses the holding/rest family');
  assertEq(confident.family, 'idle_grounded', 'confident verb reuses the grounded family');
})();

(async function dispatchesFutureBusIntent() {
  const conductor = makeConductor();
  const avatarState = makeAvatarState();
  const router = new CompanionAnimationRouter({ conductor, avatarState, logger: { warn() {} } });
  await router.onRuntimeBusEvent({
    topic: 'behavior.animation_intent',
    payload: {
      roles: ['celebrate'],
      emotion: { warmth: 0.8 },
      pose_verb: 'confident',
      explicit: true,
    },
  });
  assertEq(conductor.playCalls.length, 1, 'bus animation intent dispatches to conductor.play');
  assertEq(conductor.playCalls[0].intent.roles, ['celebrate'], 'bus intent keeps roles');
  assert(conductor.playCalls[0].options.explicit === true, 'bus explicit flag reaches conductor');
  assertEq(avatarState.animator.poseCalls, ['idle_grounded'], 'bus animation intent can also set pose family');
})();

(async function dispatchesAutonomousVerbAnimationIntent() {
  const conductor = makeConductor();
  const avatarState = makeAvatarState();
  const router = new CompanionAnimationRouter({ conductor, avatarState, logger: { warn() {} } });
  await router.onRuntimeBusEvent({
    topic: 'behavior.animation_intent',
    payload: {
      roles: ['think', 'ponder'],
      emotion: { focus: 0.9, energy: 0.36 },
      pose_verb: 'thinking',
      source: 'verb:embody_event:action:creation',
      priority: 'situational',
    },
  });
  assertEq(conductor.playCalls[0].intent.roles, ['think', 'ponder'],
    'autonomous verb animation intents route through conductor roles');
  assertEq(conductor.playCalls[0].options, { explicit: false },
    'autonomous verb animation intents remain passive');
  assertEq(avatarState._companionPoseIntent.source, 'verb:embody_event:action:creation',
    'autonomous verb animation intent source reaches pose arbitration');
  assertEq(avatarState.animator.poseCalls, ['thinking'],
    'autonomous verb animation intents reuse pose verbs');
})();

(async function dispatchesReachOutSituation() {
  const conductor = makeConductor();
  const avatarState = makeAvatarState();
  const router = new CompanionAnimationRouter({ conductor, avatarState, logger: { warn() {} } });
  await router.onRuntimeBusEvent({ topic: 'initiative.surfaced', payload: {} });
  assertEq(conductor.playCalls[0].intent.roles, ['attention-seek', 'reach-out'],
    'initiative surfaced maps to reach-out roles');
  assertEq(avatarState._companionPoseIntent.family, 'idle_engaged',
    'initiative surfaced sets an attentive pose family');
})();

(async function dispatchesToolCallAndActivitySituations() {
  const conductor = makeConductor();
  const avatarState = makeAvatarState();
  const router = new CompanionAnimationRouter({ conductor, avatarState, logger: { warn() {} } });
  await router.onRuntimeBusEvent({ topic: 'voice.tool_call', payload: {} });
  await router.onRuntimeBusEvent({
    topic: 'behavior.activity_chosen',
    payload: { kind: 'observation' },
  });
  assertEq(conductor.playCalls[0].intent.roles, ['think', 'ponder'],
    'tool call maps to thinking roles');
  assertEq(conductor.playCalls[1].intent.roles, ['listen', 'attentive', 'idle-shift'],
    'observation activity maps to attentive listening roles');
  assertEq(avatarState.animator.poseCalls, ['thinking', 'idle_lefthip'],
    'tool and observation situations update reusable pose verbs');
})();

(async function dispatchesAffectFocusAndReturnSituations() {
  const conductor = makeConductor();
  const avatarState = makeAvatarState();
  const router = new CompanionAnimationRouter({ conductor, avatarState, logger: { warn() {} } });
  await router.onRuntimeBusEvent({ topic: 'affect.changed', payload: { tag: 'frustrated' } });
  await router.onRuntimeBusEvent({ topic: 'focus.transition', payload: { to: 'user' } });
  await router.onRuntimeBusEvent({ topic: 'channel.exited', payload: {} });
  assertEq(conductor.playCalls[0].intent.roles, ['react-negative', 'angry', 'mirror-anger'],
    'frustrated affect maps to negative reaction roles');
  assertEq(conductor.playCalls[1].intent.roles, ['listen', 'attentive', 'idle-shift'],
    'focus on user maps to attentive roles');
  assertEq(conductor.playCalls[2].intent.roles, ['greet', 'wave'],
    'channel return maps to greeting roles');
  assertEq(avatarState.animator.poseCalls, ['closed', 'idle_engaged'],
    'affect/focus situations shift pose families without replaying same family');
})();

(async function audioBusMusicPinsHostPoseWithoutDoublePlayingDance() {
  const conductor = makeConductor();
  const avatarState = makeAvatarState();
  const router = new CompanionAnimationRouter({ conductor, avatarState, logger: { warn() {} } });
  await router.onAudioBusState(
    { activeTiers: ['ambient'], activeKinds: ['music'] },
    { audioRole: 'host' },
  );
  assertEq(conductor.playCalls.length, 0,
    'audio music leaves clip selection to the existing dance loop');
  assertEq(avatarState._companionPoseIntent.family, 'idle_grounded',
    'audio music pins the hosting pose family');
  assertEq(avatarState._companionPoseIntent.source, 'audio:music:host',
    'audio music records reusable audio source');
})();

(async function audioBusNarrationUsesListeningBody() {
  const conductor = makeConductor();
  const avatarState = makeAvatarState();
  const router = new CompanionAnimationRouter({ conductor, avatarState, logger: { warn() {} } });
  await router.onAudioBusState({ activeTiers: ['media'], activeKinds: ['narration'] });
  assertEq(conductor.playCalls[0].intent.roles, ['listen', 'attentive', 'idle-shift'],
    'audio narration reuses attentive listening roles');
  assertEq(avatarState._companionPoseIntent.family, 'idle_lefthip',
    'audio narration sets media attention pose');
})();

(function audioBusSpeechPinsAndReleasesTalkingPose() {
  const conductor = makeConductor();
  const avatarState = makeAvatarState();
  const router = new CompanionAnimationRouter({ conductor, avatarState, logger: { warn() {} } });
  router.onAudioBusState({ activeTiers: ['speech'], activeKinds: ['speech'] });
  assertEq(avatarState._companionPoseIntent.family, 'talking',
    'audio speech pins talking pose for lipsync');
  assertEq(avatarState._companionPoseIntent.expiresAt, 0,
    'audio speech pose lasts until audio bus releases');
  router.onAudioBusState({ activeTiers: [], activeKinds: [] });
  assertEq(avatarState._companionPoseIntent, null,
    'audio release clears audio-owned pose');
})();

(function audioBusReleaseDoesNotClearNonAudioPose() {
  const conductor = makeConductor();
  const avatarState = makeAvatarState();
  const router = new CompanionAnimationRouter({ conductor, avatarState, logger: { warn() {} } });
  router.onRuntimeBusEvent({ topic: 'initiative.surfaced', payload: {} });
  const before = avatarState._companionPoseIntent;
  router.onAudioBusState({ activeTiers: [], activeKinds: [] });
  assertEq(avatarState._companionPoseIntent, before,
    'audio release does not clear non-audio companion poses');
})();

(function audioBusDoesNotInterruptConversationPose() {
  const conductor = makeConductor();
  const avatarState = makeAvatarState();
  const router = new CompanionAnimationRouter({ conductor, avatarState, logger: { warn() {} } });
  router.onPttStateChange('speaking');
  router.onAudioBusState(
    { activeTiers: ['ambient'], activeKinds: ['music'] },
    { audioRole: 'host' },
  );
  assertEq(avatarState._companionPoseIntent.family, 'talking',
    'conversation pose keeps ownership while music starts');
  assertEq(avatarState._companionPoseIntent.source, 'ptt:speaking',
    'music does not replace the active conversation pose source');
  assertEq(avatarState.animator.poseCalls, ['talking'],
    'rejected lower-priority audio does not call the animator');
  router.onPttStateChange('idle');
  router.onAudioBusState(
    { activeTiers: ['ambient'], activeKinds: ['music'] },
    { audioRole: 'host' },
  );
  assertEq(avatarState._companionPoseIntent.source, 'audio:music:host',
    'rejected audio can claim posture after conversation releases');
  assertEq(avatarState.animator.poseCalls, ['talking', 'idle_grounded'],
    'music posture resumes once conversation no longer owns the body');
})();

(async function explicitPoseOverridesAudioAndSurvivesIdleRelease() {
  const conductor = makeConductor();
  const avatarState = makeAvatarState();
  const router = new CompanionAnimationRouter({ conductor, avatarState, logger: { warn() {} } });
  router.onAudioBusState(
    { activeTiers: ['ambient'], activeKinds: ['music'] },
    { audioRole: 'host' },
  );
  await router.onRuntimeBusEvent({
    topic: 'behavior.animation_intent',
    payload: { pose_verb: 'boundary', explicit: true },
  });
  router.onAudioBusState({ activeTiers: ['media'], activeKinds: ['narration'] });
  router.onPttStateChange('idle');
  assertEq(avatarState._companionPoseIntent.family, 'closed',
    'explicit pose overrides audio and keeps ownership');
  assertEq(avatarState._companionPoseIntent.priority, 'explicit',
    'explicit pose records the top priority tier');
  assertEq(avatarState.animator.poseCalls, ['idle_grounded', 'closed'],
    'lower-priority narration and idle release do not replay or clear explicit posture');
})();

(async function throttlesRepeatedSituationalActions() {
  const conductor = makeConductor();
  let now = 1000;
  const router = new CompanionAnimationRouter({
    conductor,
    logger: { warn() {} },
    now: () => now,
  });
  await router.onRuntimeBusEvent({ topic: 'voice.tool_call', payload: {} });
  now += 1000;
  await router.onRuntimeBusEvent({ topic: 'voice.tool_call', payload: {} });
  now += 6000;
  await router.onRuntimeBusEvent({ topic: 'voice.tool_call', payload: {} });
  assertEq(conductor.playCalls.length, 2,
    'situational actions throttle repeated same-key events');
})();

(async function speakingSuppressesPassiveButNotExplicitActions() {
  const conductor = makeConductor();
  const avatarState = makeAvatarState();
  const router = new CompanionAnimationRouter({ conductor, avatarState, logger: { warn() {} } });
  router.onPttStateChange('speaking');
  await router.onRuntimeBusEvent({ topic: 'initiative.surfaced', payload: {} });
  await router.onRuntimeBusEvent({
    topic: 'behavior.animation_intent',
    payload: { id: 'peace-sign', pose_verb: 'confident', explicit: true },
  });
  assertEq(conductor.playCalls.length, 0, 'speaking suppresses passive situation actions');
  assertEq(conductor.playByIdCalls[0], { id: 'peace-sign', options: { explicit: true } },
    'explicit animation intent can still route while speaking');
  assertEq(avatarState.animator.poseCalls, ['talking', 'idle_grounded'],
    'speaking pose is active and explicit pose intent can override it');
})();

(function pttStateFeedsPresenceAndListeningPose() {
  const conductor = makeConductor();
  const avatarState = makeAvatarState();
  const states = [];
  const router = new CompanionAnimationRouter({
    conductor,
    avatarState,
    logger: { warn() {} },
    hooks: { onStateChange: (state) => states.push(state) },
    now: () => 1234,
  });
  router.onPttStateChange('recording');
  assertEq(states, ['recording'], 'recording state fans out to avatar presence');
  assertEq(conductor.playCalls[0].intent.roles, ['listen', 'attentive', 'idle-shift'],
    'recording reuses pose trigger listening intent');
  assertEq(avatarState._companionPoseIntent.family, 'idle_engaged',
    'recording sets listening pose intent for the avatar loop');
})();

(function pttStateSetsThinkingAndSpeakingPoseFamilies() {
  const conductor = makeConductor();
  const avatarState = makeAvatarState();
  const router = new CompanionAnimationRouter({
    conductor,
    avatarState,
    logger: { warn() {} },
    now: () => 2000,
  });
  router.onPttStateChange('processing');
  router.onPttStateChange('speaking');
  assertEq(avatarState.animator.poseCalls, ['thinking', 'talking'],
    'processing and speaking map into existing pose families');
  assertEq(avatarState._companionPoseIntent.expiresAt, 8500,
    'speaking pose intent stores an expiry for avatar.js recovery');
})();

(async function explicitPoseOnlyIntentDoesNotNeedConductorRoles() {
  const conductor = makeConductor();
  const avatarState = makeAvatarState();
  const router = new CompanionAnimationRouter({
    conductor,
    avatarState,
    logger: { warn() {} },
  });
  await router.onRuntimeBusEvent({
    topic: 'behavior.animation_intent',
    payload: { pose_verb: 'boundary', explicit: true },
  });
  assertEq(conductor.playCalls.length, 0, 'pose-only intent does not call conductor.play');
  assertEq(avatarState.animator.poseCalls, ['closed'], 'pose-only intent updates pose family');
})();

(function transcriptFeedsPresenceAndExplicitPoseTrigger() {
  const conductor = makeConductor();
  const transcripts = [];
  const router = new CompanionAnimationRouter({
    conductor,
    logger: { warn() {} },
    hooks: { onUserTranscript: (text, final) => transcripts.push({ text, final }) },
  });
  router.onTranscript('please do a peace sign', { final: true });
  assertEq(transcripts, [{ text: 'please do a peace sign', final: true }],
    'final transcript fans out to avatar presence');
  assertEq(conductor.playByIdCalls[0], { id: 'peace-sign', options: {} },
    'explicit transcript request reuses pose trigger playById');
})();

(function duplicateFinalTranscriptDoesNotReplayPose() {
  const conductor = makeConductor();
  const router = new CompanionAnimationRouter({
    conductor,
    logger: { warn() {} },
    now: () => 1000,
  });
  router.onTranscript('wave hello', { final: true });
  router.onTranscript('wave hello', { final: true });
  assertEq(conductor.playByIdCalls.length, 1,
    'duplicate final transcripts do not replay explicit poses');
})();

(function llmAndTtsSignalsReusePresenceAndPoseTrigger() {
  const conductor = makeConductor();
  const deltas = [];
  const router = new CompanionAnimationRouter({
    conductor,
    logger: { warn() {} },
    hooks: { onLLMDelta: (text) => deltas.push(text) },
  });
  router.onLLMDelta('That is wonderful.');
  router.onTtsStart('That is wonderful.');
  router.onTtsEnd();
  assertEq(deltas, ['That is wonderful.'], 'LLM deltas fan out to semantic presence');
  assertEq(conductor.playCalls[0].intent.roles, ['emphasize', 'react-positive', 'excitement-peak'],
    'TTS sentence sentiment reuses pose trigger response intent');
})();

setTimeout(() => {
  console.log(`\n${_ran - _failed}/${_ran} assertions passed`);
  if (_failed > 0) process.exit(1);
}, 0);
