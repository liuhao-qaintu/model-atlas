import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import App from './App.vue'
import './style.css'
import './product.css'
import { scrollPosition } from './navigation'
import { account } from './auth.js'
import { api } from './api.js'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path:'/login',meta:{title:'登录',public:true},component:()=>import('./views/AuthView.vue') },
    { path:'/register',meta:{title:'注册',public:true},component:()=>import('./views/AuthView.vue') },
    { path:'/account',meta:{title:'账号设置'},component:()=>import('./views/AuthView.vue') },
    { path: '/', meta: { title: '首页' }, component: () => import('./views/HomeView.vue') },
    { path: '/learn/autoencoder', meta: { title: 'AE 阅读路线' }, component: () => import('./views/LearningExampleView.vue') },
    { path: '/models/:modelId?', meta: { title: '模型学习' }, component: () => import('./views/LibraryView.vue') },
    { path: '/graph', meta: { title: '模型关系' }, component: () => import('./views/GraphView.vue') },
    { path: '/chat', meta: { title: '智能问答' }, component: () => import('./views/ChatView.vue') },
    { path: '/research', meta: { title: '比较研究' }, component: () => import('./views/ResearchView.vue') },
    { path: '/sources', redirect: '/documents' },
    { path: '/documents/:documentId?', meta: { title: '我的文献' }, component: () => import('./views/DocumentsView.vue') },
    { path: '/analysis', meta: { title: '资料分析' }, component: () => import('./views/AnalysisView.vue') },
    { path: '/manage', meta: { title: '模型管理' }, component: () => import('./views/ManageView.vue') },
    { path: '/evaluation', meta: { title: '质量评测' }, component: () => import('./views/EvaluationView.vue') },
    { path: '/:pathMatch(.*)*', redirect: '/models' },
  ],
  scrollBehavior: scrollPosition,
})
let checkedAccount = false
router.beforeEach(async to => {
  if (to.meta.public) return true
  if (!checkedAccount) {
    try { account.value = await api('/auth/me') } catch { account.value = null }
    checkedAccount = true
  }
  if (!account.value) return {path:'/login',query:{next:to.fullPath}}
  if (['/analysis','/evaluation'].some(path=>to.path===path || to.path.startsWith(path+'/')) && !account.value.is_maintainer) return '/'
  return true
})
window.addEventListener('atlas-unauthorized',()=>router.replace('/login'))
createApp(App).use(router).mount('#app')
