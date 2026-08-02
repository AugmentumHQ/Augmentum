/* ==========================================================================
   Analytical phase constants — single source shared by flow-bar and
   analytical reasoning-panel modules.  Backend phase keys live in
   augmentum/modes/analytical/prompts.py — keep in sync.
   ========================================================================== */

export const PHASE_NAMES = {
  ASSESS: 'Assess',
  SEARCH: 'Search',
  GATHER: 'Gather',
  IDENTIFY: 'Identify',
  RELEVANT: 'Research',
  APPLY: 'Analyze',
  VERIFY: 'Verify',
  RESPOND: 'Respond',
  CONCLUDE: 'Conclude',
};

export const PHASE_DESCRIPTIONS = {
  ASSESS: 'Evaluating query complexity and intent',
  SEARCH: 'Searching for relevant information',
  GATHER: 'Collecting data from multiple sources',
  IDENTIFY: 'Identifying key concepts and entities',
  RELEVANT: 'Researching related context and sources',
  APPLY: 'Analyzing findings and forming conclusions',
  VERIFY: 'Cross-checking facts and validating results',
  RESPOND: 'Synthesizing a comprehensive response',
  CONCLUDE: 'Finalizing and formatting the answer',
};
