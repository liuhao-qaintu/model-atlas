import { computed, onMounted, ref } from 'vue'
import { api } from './api'
import { aePaper } from './aeJourney'

export function useAeJourney() {
  const documents = ref([]), error = ref(''), loaded = ref(false)
  onMounted(async () => {
    try { documents.value = await api('/documents') }
    catch (e) { error.value = e.message }
    finally { loaded.value = true }
  })
  return { paper: computed(() => aePaper(documents.value)), error, loaded }
}
