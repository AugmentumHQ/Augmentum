/* ==========================================================================
   Augmentum — Chat Module (shim)
   Re-exports from chat/index.js which contains the full implementation.
   ========================================================================== */

export {
  chat,
  initChat,
  icons,
  sessionStore,
  MessageRenderer,
  ChatInput,
  ChatStream,
  renderMarkdown,
  highlightCode,
  TtsSentenceBuffer, TtsAudioPipeline, ttsProgressiveFeed, ttsProgressiveFinish,
  ttsProgressiveCancel, ttsStopCurrent, ttsChatWarmup, ttsCleanText, ttsSplitChunks,
  ttsFetchAudio, ttsPlayBlob, ttsPlayMessage, ttsQueueAutoRead, ttsProcessQueue,
  setActiveSessionGetter, setCharacterVoiceLookup,
  initCodeBlockActions, closePreviewModal, restoreCodeVersions,
  toggleHtmlPreview, runPythonCode, toggleSvgPreview, toggleCodeEdit,
  downloadCodeBlock, showVersion, getVersionIdx, getBlock,
  updateBlock, hydrateCodeBlocks, regenerateMarkdown, getSessionNode,
  updateVersionIndicator, getBlocksForMessage, getOutputPanel,
  appendOutput, appendError, renderExecutionOutput, getPythonPreamble,
  codeMindValidate, getPendingBlockEdits, clearPendingBlockEdits,
  getFixRetryCount,
  showAskAiPrompt, executeAiEdit, showQuickActionsMenu,
  autoFixCodeBlock, silentLint, applyDiffPatches,
  createStreamingDiff, parseMultiBlockResponse,
  computeLineDiff, renderDiffLines,
  fixHTML, fixCSS, fixPython, fixJSON,
  initIllustrateMoment,
  initMicButton,
  memGlowInit, memGlowRecalling, memGlowIdle, memGlowLearned,
  memStartPolling, memStopPolling, memMarkExtracting,
  scanLorebook,
  renderYouTubeCards,
  renderToolDeliverable,
  showGeneratedImage,
  fetchAndShowLatestImage,
  openImageLightbox,
  surfaceRelatedFiles,
  saveCodeBlockToLibrary,
} from './chat/index.js';

// Re-export tree.js wildcard (covers buildMessagesForAPI, etc.)
export * from './chat/tree.js';
