<script setup>
import { computed,reactive,ref } from 'vue'
import { useRoute,useRouter } from 'vue-router'
import { api } from '../api.js'
import { account,safeNext } from '../auth.js'

const route=useRoute(),router=useRouter()
const registering=computed(()=>route.path==='/register')
const settings=computed(()=>route.path==='/account')
const title=computed(()=>settings.value?'修改密码':registering.value?'创建账号':'登录 Model Atlas')
const form=reactive({username:'',password:'',confirmation:'',display_name:'',old_password:''})
const busy=ref(false),error=ref(''),message=ref('')
async function submit(){
  if(busy.value) return
  error.value='';message.value=''
  if((registering.value || settings.value) && form.password!==form.confirmation){error.value='两次输入的密码不一致';return}
  busy.value=true
  try {
    if(settings.value){
      await api('/auth/password',{old_password:form.old_password,new_password:form.password})
      account.value=null;window.location.assign('/login')
    } else if(registering.value){
      await api('/auth/register',{username:form.username,password:form.password,display_name:form.display_name})
      form.password='';form.confirmation='';message.value='注册成功，请登录'
      await router.replace('/login')
    } else {
      account.value=await api('/auth/login',{username:form.username,password:form.password})
      window.location.assign(safeNext(route.query.next))
    }
  } catch(problem){error.value=problem.message}
  finally{busy.value=false}
}
</script>

<template>
  <section :class="['atlas-auth-page',{'account-page':settings}]">
    <div class="atlas-auth-card">
      <RouterLink class="auth-brand" to="/">Model Atlas</RouterLink>
      <h1>{{ title }}</h1>
      <p class="auth-intro">{{ settings?'修改后需要重新登录，原登录状态将失效。':'阅读模型知识、管理个人文献与会话，并查看回答依据。' }}</p>
      <form @submit.prevent="submit">
        <template v-if="!settings">
          <label for="atlas-username">用户名</label>
          <input id="atlas-username" v-model.trim="form.username" required minlength="3" maxlength="40" pattern="[A-Za-z0-9_.-]+" autocomplete="username" placeholder="3—40位英文、数字、点、下划线或连字符" />
          <template v-if="registering"><label for="atlas-name">显示名称（选填）</label><input id="atlas-name" v-model.trim="form.display_name" maxlength="80" autocomplete="nickname" /></template>
        </template>
        <template v-else><label for="atlas-old-password">原密码</label><input id="atlas-old-password" v-model="form.old_password" type="password" required autocomplete="current-password" /></template>
        <label for="atlas-password">{{ settings?'新密码':'密码' }}</label>
        <input id="atlas-password" v-model="form.password" type="password" required :minlength="registering || settings?8:undefined" :autocomplete="registering || settings?'new-password':'current-password'" />
        <template v-if="registering || settings"><label for="atlas-confirm">确认密码</label><input id="atlas-confirm" v-model="form.confirmation" type="password" required autocomplete="new-password" /><p class="auth-hint">密码至少8个字符，且不超过72个UTF-8字节。</p></template>
        <p v-if="error" role="alert" class="auth-error">{{ error }}</p><p v-if="message" role="status">{{ message }}</p>
        <button class="button primary auth-submit" :disabled="busy" type="submit">{{ busy?'正在处理…':settings?'修改并重新登录':registering?'注册':'登录' }}</button>
      </form>
      <p v-if="!settings" class="auth-switch"><RouterLink :to="registering?'/login':'/register'">{{ registering?'已有账号，返回登录':'没有账号？创建账号' }}</RouterLink></p>
    </div>
  </section>
</template>

<style scoped>
.atlas-auth-page{min-height:100vh;display:grid;place-items:center;padding:24px;background:#f4f7fa}.account-page{min-height:70vh;background:transparent}.atlas-auth-card{width:min(460px,100%);padding:36px;background:white;border:1px solid var(--line);border-radius:16px;box-shadow:0 16px 48px #163b5510}.auth-brand{font-size:25px;font-weight:750;color:var(--green)}h1{font-size:24px;margin:18px 0 10px}.auth-intro,.auth-hint,.auth-switch{font-size:13px;color:var(--muted);line-height:1.8}.auth-intro{margin-bottom:22px}form label{display:block;font-size:14px;margin:16px 0 8px}form input{width:100%;padding:11px 12px;border:1px solid var(--line);border-radius:7px;font:inherit;background:white}.auth-submit{width:100%;margin:22px 0 10px;justify-content:center}.auth-error{color:#ab3939;font-size:13px;line-height:1.7;margin-top:14px}.auth-switch{margin-top:14px;text-align:center}.auth-hint{margin-top:10px}
</style>
