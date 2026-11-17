// Actual frontend + isolated API, synthetic provider bytes. No quality claims.
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';
const evidence=path.resolve('scratch/image-browser');
fs.mkdirSync(evidence,{recursive:true});
const browser=await chromium.launch({headless:true});
const context=await browser.newContext();
const page=await context.newPage();
const passed=[];
const events=[];
page.on('request',r=>events.push(r.method()+' '+r.url()));
page.on('pageerror',e=>events.push('PAGE ERROR '+e.message));
const check=(name,value)=>{if(!value)throw new Error(name);passed.push(name);};
try {
  // Test login route is outside API prefix; set cookie through loopback backend.
  const login=await context.request.get('http://127.0.0.1:8319/__contract__/login/alice');
  check('synthetic fixture login',login.ok());
  await page.goto('http://127.0.0.1:3319',{waitUntil:'domcontentloaded'});
  const splash=page.locator('.jru-initial-splash-overlay');
  if(await splash.isVisible().catch(()=>false))await splash.click();
  await page.getByText('Synthetic',{exact:true}).waitFor();
  fs.writeFileSync(path.join(evidence,'initial-text.txt'),await page.locator('body').innerText());
  const studio=page.getByRole('button',{name:/vision studio|images/i}).first();
  await studio.click();
  await page.locator('#image-prompt-input').waitFor();
  const generate=page.locator('form').filter({has:page.locator('#image-prompt-input')}).getByRole('button',{name:/generate|synthesize/i});
  async function create(prompt,style='photorealistic') {
    await page.locator('#image-prompt-input').fill(prompt);
    await page.locator('#image-style-select').selectOption(style);
    const [result]=await Promise.all([
      page.waitForResponse(r=>r.url().endsWith('/api/v1/image/generate')&&r.request().method()==='POST'),
      generate.click()
    ]);
    return [result,await result.json()];
  }
  let saved;
  for(const [prompt,style] of [['a boat','photorealistic'],['a realistic sunset landscape','photorealistic'],['an anime illustration','anime']]) {
    const [response,result]=await create(prompt,style);
    check('render fixture: '+prompt,response.ok()&&result.success&&result.model.includes('synthetic browser'));
    saved=result;
    await page.waitForFunction(url=>Array.from(document.images).some(img=>img.getAttribute('src')===url&&img.complete&&img.naturalWidth>0),result.image_url);
    const close=page.locator('.modal-close-btn').first();
    await close.waitFor();
    await close.click();
  }
  await page.screenshot({path:path.join(evidence,'gallery.png'),fullPage:true});
  const timeoutPrompt='timeout fixture '+Date.now();
  const [timeout]=await create(timeoutPrompt);
  check('timeout shown',timeout.status()===503);
  check('timeout message preserved',await page.getByText(/generation timed out/i).isVisible());
  const [retry,retried]=await create(timeoutPrompt);
  check('retry works',retry.ok()&&retried.success);
  const close=page.locator('.modal-close-btn').first();
  await close.waitFor();
  await close.click();
  const [quota]=await create('quota fixture');
  check('quota response',quota.status()===429);
  check('quota message shown',await page.getByText(/daily image quota is exhausted/i).isVisible());
  const content=await context.request.get('http://127.0.0.1:3319'+saved.image_url);
  check('previous image opens',content.ok()&&content.headers()['content-type']==='image/png');
  const other=await browser.newContext();
  await other.request.get('http://127.0.0.1:8319/__contract__/login/bob');
  check('foreign download denied',(await other.request.get('http://127.0.0.1:3319'+saved.image_url)).status()===404);
  check('foreign delete denied',(await other.request.delete('http://127.0.0.1:3319/api/v1/image/'+saved.image_id)).status()===404);
  check('owned deletion',(await context.request.delete('http://127.0.0.1:3319/api/v1/image/'+saved.image_id)).ok());
  check('deleted image unavailable',(await context.request.get('http://127.0.0.1:3319'+saved.image_url)).status()===404);
  await other.close();
  await page.getByRole('button',{name:/back to chat/i}).click();
  await page.locator('textarea[aria-label="Message input"]').fill('generate image of boat');
  const [chat]=await Promise.all([
    page.waitForResponse(r=>r.url().endsWith('/api/v1/chat')&&r.request().method()==='POST'),
    page.getByRole('button',{name:'Send message',exact:true}).click()
  ]);
  check('chat streaming response',chat.ok());
  const imageCard=page.locator('.generated-image-img').last();
  await imageCard.waitFor();
  await page.waitForFunction(()=>Array.from(document.querySelectorAll('.generated-image-img')).some(img=>img.complete&&img.naturalWidth===256));
  check('chat renders owned image',(await imageCard.getAttribute('src')).includes('/api/v1/image/'));
  await page.screenshot({path:path.join(evidence,'chat.png'),fullPage:true});
  fs.writeFileSync(path.join(evidence,'results.json'),JSON.stringify({mode:'synthetic provider contract; actual browser and backend',passed,actual_inference:false},null,2));
  console.log(JSON.stringify({passed:passed.length,checks:passed,actual_inference:false}));
} catch(error) {
  fs.writeFileSync(path.join(evidence,'events.json'),JSON.stringify(events,null,2));
  fs.writeFileSync(path.join(evidence,'failure-text.txt'),await page.locator('body').innerText());
  await page.screenshot({path:path.join(evidence,'failure.png'),fullPage:true});
  console.error(error);
  process.exitCode=1;
} finally {await browser.close();}
