export function shouldSendOnEnter(event) {
  return event.key === 'Enter' && !event.shiftKey && !event.altKey && !event.isComposing && event.keyCode !== 229
}
