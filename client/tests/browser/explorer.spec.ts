import { test, expect } from "@playwright/test";
const taskId = (i: number) => i.toString(16).padStart(64,"0");

test.beforeAll(async ({request}) => {
  // Refuse to run mutation tests against a real extraction run.
  const health=await request.get("/api/health");
  expect(health.ok()).toBeTruthy();
  expect((await health.json()).dataset).toBe("browser-fixture");
});

test("all-cohort browsing, pagination, search, detail and back navigation", async ({page}) => {
  await page.goto("/");
  await expect(page.getByRole("heading",{name:"Explore discovered interactions"})).toBeVisible();
  await expect(page.getByText("62 matching contexts")).toBeVisible();
  await expect(page.locator("tbody tr")).toHaveCount(50);
  await page.screenshot({path:"outputs/browser-overview.png"});
  await page.getByRole("button",{name:"Next page"}).click();
  await expect(page).toHaveURL(/page=2/);
  await expect(page.locator("tbody tr")).toHaveCount(12);
  await page.getByLabel("Cohort",{exact:true}).selectOption("beyond_pilot");
  await page.getByRole("button",{name:"Search",exact:true}).click();
  await expect(page.getByText("10 matching contexts")).toBeVisible();
  await expect(page.locator("tbody tr")).toHaveCount(10);
  await page.getByRole("link",{name:/View evidence/}).first().click();
  await expect(page.getByRole("heading",{name:"Supporting evidence",exact:true})).toBeVisible();
  await expect(page.getByRole("heading",{name:"Provenance",exact:true})).toBeVisible();
  await expect(page.getByText("Quote verified when classified")).toBeVisible();
  await page.getByRole("link",{name:"Back to results",exact:true}).click();
  await expect(page).toHaveURL(/cohort=beyond_pilot/);
  await expect(page.getByText("10 matching contexts")).toBeVisible();
  await page.getByLabel("Gene",{exact:true}).fill("absent");
  await page.getByRole("button",{name:"Search",exact:true}).click();
  await expect(page.getByRole("heading",{name:"No contexts match this view"})).toBeVisible();
});

test("direct lookup, cached source, complete provenance and JSON download", async ({page}) => {
  await page.goto(`/contexts/${taskId(1)}`);
  await expect(page.getByText("Hashes verified",{exact:true})).toBeVisible();
  await expect(page.locator(".source-text")).toContainText("Venetoclax inhibits BCL2");
  await expect(page.getByRole("heading",{name:"Detected entity mentions"})).toBeVisible();
  const download=page.waitForEvent("download");
  await page.getByRole("button",{name:"Provenance JSON"}).click();
  expect((await download).suggestedFilename()).toContain("provenance.json");
  await page.getByText("Complete saved classifier result",{exact:true}).click();
  await expect(page.locator("pre").filter({hasText:'"interaction": true'})).toBeVisible();
  await page.goto(`/contexts/${taskId(63)}`);
  await expect(page.getByText(/Historical entry:/)).toBeVisible();
});

test("explicit retrieval success and changed-source outcome", async ({page}) => {
  let posts=0;
  await page.route(`**/api/contexts/${taskId(53)}/source`, async route => {
    if(route.request().method()!=="POST") return route.continue();
    posts++;
    await route.fulfill({json:{state:"verified",origin:"pmc_retrieval",context:"Verified retrieved source text.",message:"Hashes match.",read_at:new Date().toISOString()}});
  });
  await page.goto(`/contexts/${taskId(53)}`);
  await expect(page.getByRole("button",{name:"Fetch source context"})).toBeEnabled();
  expect(posts).toBe(0);
  await page.getByRole("button",{name:"Fetch source context"}).click();
  await expect(page.locator(".source-text")).toContainText("Verified retrieved source text.");
  expect(posts).toBe(1);
  await page.route(`**/api/contexts/${taskId(54)}/source`, async route => {
    if(route.request().method()!=="POST") return route.continue();
    await route.fulfill({json:{state:"hash_mismatch",origin:"pmc_retrieval",context:null,message:"Available text does not match the saved section and context hashes.",read_at:new Date().toISOString()}});
  });
  await page.goto(`/contexts/${taskId(54)}`);
  await page.getByRole("button",{name:"Fetch source context"}).click();
  await expect(page.getByText("Available text does not match the saved section and context hashes.")).toBeVisible();
  await expect(page.locator(".source-text")).toHaveCount(0);
});

test("missing entry and narrow-screen navigation", async ({page}) => {
  await page.goto(`/contexts/${taskId(999)}`);
  await expect(page.getByRole("heading",{name:"Context unavailable"})).toBeVisible();
  await page.setViewportSize({width:390,height:844});
  await page.goto("/");
  await expect(page.getByText("62 matching contexts")).toBeVisible();
  await expect(page.getByRole("button",{name:"Search",exact:true})).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
});

test("dgiLIT Claim Explorer branding and shared rating interactions", async ({page,request}) => {
  await page.goto(`/contexts/${taskId(1)}`);
  await expect(page).toHaveTitle("dgiLIT · Claim Explorer");
  await expect(page.getByRole("link",{name:"dgiLIT home"})).toContainText("Claim Explorer");
  await request.put(`/api/contexts/${taskId(1)}/rating`,{data:{rating:null}});
  await page.reload();
  const up=page.getByRole("button",{name:"Thumbs up",exact:true});
  const down=page.getByRole("button",{name:"Thumbs down",exact:true});
  await expect(up).toHaveAttribute("aria-pressed","false");
  await up.click();
  await expect(up).toHaveAttribute("aria-pressed","true");
  await expect(down).toHaveAttribute("aria-pressed","false");
  await page.reload();
  await expect(up).toHaveAttribute("aria-pressed","true");
  await down.click();
  await expect(down).toHaveAttribute("aria-pressed","true");
  await expect(up).toHaveAttribute("aria-pressed","false");
  await down.click();
  await expect(down).toHaveAttribute("aria-pressed","false");
  await expect(page.getByRole("status")).toContainText("No assessment yet");
  await up.click();
  await expect(up).toHaveAttribute("aria-pressed","true");
  const download=page.waitForEvent("download");
  await page.getByRole("button",{name:"Provenance JSON"}).click();
  const file=await (await download).path();
  const {readFile}=await import("node:fs/promises");
  const record=JSON.parse(await readFile(file!,"utf8"));
  expect(record.human_assessment.rating).toBe("up");
  expect(record.task.result.interaction).toBe(true);
  // A new browser context also reads the shared assessment.
  const other=await page.context().browser()!.newContext();
  const otherPage=await other.newPage();
  await otherPage.goto(`${new URL(page.url()).origin}/contexts/${taskId(1)}`);
  await expect(otherPage.getByRole("button",{name:"Thumbs up",exact:true})).toHaveAttribute("aria-pressed","true");
  await other.close();
});

test("ratings show badges on compact rows including mobile", async ({page,request}) => {
  await request.put(`/api/contexts/${taskId(59)}/rating`,{data:{rating:"up"}});
  await request.put(`/api/contexts/${taskId(58)}/rating`,{data:{rating:"down"}});
  await page.goto("/?cohort=beyond_pilot");
  const upRow=page.locator("tbody tr").filter({has:page.locator(`a[href*="${taskId(59)}"]`)});
  const downRow=page.locator("tbody tr").filter({has:page.locator(`a[href*="${taskId(58)}"]`)});
  await expect(upRow.getByLabel("Shared assessment: thumbs up")).toBeVisible();
  await expect(downRow.getByLabel("Shared assessment: thumbs down")).toBeVisible();
  await expect(upRow.getByLabel("Shared assessment: thumbs up")).toHaveText("");
  await expect(downRow.getByLabel("Shared assessment: thumbs down")).toHaveText("");
  await page.setViewportSize({width:390,height:844});
  for(const [row,label] of [[upRow,"Shared assessment: thumbs up"],[downRow,"Shared assessment: thumbs down"]] as const){
    const badge=row.getByLabel(label);await badge.scrollIntoViewIfNeeded();
    const bounds=await badge.boundingBox();
    expect(bounds!.x).toBeGreaterThanOrEqual(0);
    expect(bounds!.x+bounds!.width).toBeLessThanOrEqual(390);
  }
  await downRow.getByRole("link",{name:/View evidence/}).click();
  await expect(page.getByRole("button",{name:"Thumbs down",exact:true})).toHaveAttribute("aria-pressed","true");
  await page.getByRole("link",{name:"Back to results",exact:true}).click();
  await expect(page).toHaveURL(/cohort=beyond_pilot/);
  await expect(page.getByLabel("Shared assessment: thumbs down")).toBeVisible();
});

test("save indicator and failed saves preserve the selected rating", async ({page,request}) => {
  await request.put(`/api/contexts/${taskId(2)}/rating`,{data:{rating:"up"}});
  await page.goto(`/contexts/${taskId(2)}`);
  const up=page.getByRole("button",{name:"Thumbs up",exact:true});
  const down=page.getByRole("button",{name:"Thumbs down",exact:true});
  await expect(up).toHaveAttribute("aria-pressed","true");
  let release: (()=>void) | undefined;
  const pending=new Promise<void>(resolve=>{release=resolve;});
  await page.route(`**/api/contexts/${taskId(2)}/rating`,async route=>{
    await pending;
    await route.fulfill({status:503,json:{detail:"The claim rating could not be saved. Your previous assessment is unchanged."}});
  });
  await down.click();
  await expect(down).toBeDisabled();
  await expect(up).toBeDisabled();
  await expect(page.getByRole("status")).toContainText("Saving assessment");
  release!();
  await expect(page.getByRole("alert")).toContainText("previous assessment is unchanged");
  await expect(up).toHaveAttribute("aria-pressed","true");
  await expect(down).toHaveAttribute("aria-pressed","false");
  await expect(down).toBeEnabled();
  expect(await (await request.get(`/api/contexts/${taskId(2)}`)).json()).toMatchObject({rating:"up"});
});

test("assessment filters include all pages and preserve search state", async ({page,request}) => {
  const list=await (await request.get('/api/contexts?page_size=100')).json();
  for(const row of list.items){
    await request.put(`/api/contexts/${row.task_id}/rating`,{data:{rating:null}});
  }
  await request.put(`/api/contexts/${taskId(1)}/rating`,{data:{rating:'up'}});
  await request.put(`/api/contexts/${taskId(53)}/rating`,{data:{rating:'up'}});
  await request.put(`/api/contexts/${taskId(59)}/rating`,{data:{rating:'down'}});
  await page.goto('/?page=2');
  await expect(page.getByText('62 matching contexts')).toBeVisible();
  await page.getByLabel('Assessment',{exact:true}).selectOption('up');
  await page.getByRole('button',{name:'Search',exact:true}).click();
  await expect(page).toHaveURL(/rating=up/);
  await expect(page).not.toHaveURL(/page=2/);
  await expect(page.getByText('2 matching contexts')).toBeVisible();
  await expect(page.locator('tbody tr')).toHaveCount(2);
  await expect(page.locator('.rating-badge.rating-up')).toHaveCount(2);
  await page.getByRole('link',{name:/View evidence/}).first().click();
  await expect(page.getByRole('button',{name:'Thumbs up',exact:true})).toHaveAttribute('aria-pressed','true');
  await page.getByRole('link',{name:'Back to results',exact:true}).click();
  await expect(page).toHaveURL(/rating=up/);
  await expect(page.getByLabel('Assessment',{exact:true})).toHaveValue('up');
  await page.reload();
  await expect(page.getByText('2 matching contexts')).toBeVisible();
  await page.getByLabel('Assessment',{exact:true}).selectOption('hide_down');
  await page.getByRole('button',{name:'Search',exact:true}).click();
  await expect(page.getByText('61 matching contexts')).toBeVisible();
  await expect(page.locator('.rating-badge.rating-down')).toHaveCount(0);
  await page.getByRole('button',{name:'Next page'}).click();
  await expect(page).toHaveURL(/rating=hide_down/);
  await expect(page.locator('tbody tr')).toHaveCount(11);
  await expect(page.locator('.rating-badge.rating-down')).toHaveCount(0);
  await page.getByLabel('Cohort',{exact:true}).selectOption('beyond_pilot');
  await page.getByRole('button',{name:'Search',exact:true}).click();
  await expect(page.getByText('9 matching contexts')).toBeVisible();
  await page.getByRole('link',{name:'Reset',exact:true}).click();
  await expect(page.getByText('62 matching contexts')).toBeVisible();
});

test("thumbs-down reasons are optional, independently editable and included in provenance", async ({page,request}) => {
  const id=taskId(3);
  await request.put(`/api/contexts/${id}/rating`,{data:{rating:null}});
  await page.goto(`/contexts/${id}`);
  const down=page.getByRole("button",{name:"Thumbs down",exact:true});
  const up=page.getByRole("button",{name:"Thumbs up",exact:true});
  const reasons=page.getByRole("group",{name:"Error reasons",exact:true});
  await expect(reasons).toHaveCount(0);
  await down.click();
  await expect(reasons).toBeVisible();
  await expect(reasons.getByRole("button")).toHaveCount(8);
  expect((await (await request.get(`/api/contexts/${id}`)).json()).error_reasons).toEqual([]);
  const gene=reasons.getByRole("button",{name:"Incorrect gene grouping",exact:true});
  const evidence=reasons.getByRole("button",{name:"Not enough evidence",exact:true});
  await gene.click();
  await expect(gene).toHaveAttribute("aria-pressed","true");
  await evidence.focus();
  await page.keyboard.press("Space");
  await expect(evidence).toHaveAttribute("aria-pressed","true");
  await page.reload();
  await expect(gene).toHaveAttribute("aria-pressed","true");
  await expect(evidence).toHaveAttribute("aria-pressed","true");
  await gene.focus();
  await page.keyboard.press("Enter");
  await expect(gene).toHaveAttribute("aria-pressed","false");
  await expect(evidence).toHaveAttribute("aria-pressed","true");
  const download=page.waitForEvent("download");
  await page.getByRole("button",{name:"Provenance JSON"}).click();
  const file=await (await download).path();
  const {readFile}=await import("node:fs/promises");
  const record=JSON.parse(await readFile(file!,"utf8"));
  expect(record.human_assessment).toMatchObject({rating:"down",error_reasons:["not_enough_evidence"]});
  const other=await page.context().browser()!.newContext();
  try {
    const otherPage=await other.newPage();
    await otherPage.goto(`${new URL(page.url()).origin}/contexts/${id}`);
    await expect(otherPage.getByRole("button",{name:"Not enough evidence",exact:true})).toHaveAttribute("aria-pressed","true");
  } finally {await other.close();}
  await up.click();
  await expect(up).toHaveAttribute("aria-pressed","true");
  await expect(reasons).toHaveCount(0);
  expect((await (await request.get(`/api/contexts/${id}`)).json()).error_reasons).toEqual([]);
  await down.click();
  await expect(evidence).toHaveAttribute("aria-pressed","false");
  await evidence.click();
  await expect(evidence).toHaveAttribute("aria-pressed","true");
  await down.click();
  await expect(reasons).toHaveCount(0);
  expect((await (await request.get(`/api/contexts/${id}`)).json()).error_reasons).toEqual([]);
});

test("failed reason saves disable controls and preserve the saved selection", async ({page,request}) => {
  const id=taskId(4);
  await request.put(`/api/contexts/${id}/rating`,{data:{rating:"down",error_reasons:["not_enough_evidence"]}});
  await page.goto(`/contexts/${id}`);
  const reasons=page.getByRole("group",{name:"Error reasons",exact:true});
  const previous=reasons.getByRole("button",{name:"Not enough evidence",exact:true});
  const next=reasons.getByRole("button",{name:"Incorrect directionality",exact:true});
  await expect(previous).toHaveAttribute("aria-pressed","true");
  let release: (()=>void) | undefined;
  const pending=new Promise<void>(resolve=>{release=resolve;});
  await page.route(`**/api/contexts/${id}/rating`,async route=>{
    await pending;
    await route.fulfill({status:503,json:{detail:"The claim rating could not be saved. Your previous assessment is unchanged."}});
  });
  await next.click();
  await expect(page.getByRole("status")).toContainText("Saving assessment");
  for(const button of await reasons.getByRole("button").all())await expect(button).toBeDisabled();
  await expect(page.getByRole("button",{name:"Thumbs up",exact:true})).toBeDisabled();
  await expect(page.getByRole("button",{name:"Thumbs down",exact:true})).toBeDisabled();
  await expect(previous).toHaveAttribute("aria-pressed","true");
  await expect(next).toHaveAttribute("aria-pressed","false");
  release!();
  await expect(page.getByRole("alert")).toContainText("previous assessment is unchanged");
  await expect(next).toBeEnabled();
  await expect(previous).toHaveAttribute("aria-pressed","true");
  await expect(next).toHaveAttribute("aria-pressed","false");
  expect((await (await request.get(`/api/contexts/${id}`)).json()).error_reasons).toEqual(["not_enough_evidence"]);
});

test("all eight reason buttons wrap and remain usable on mobile", async ({page,request}) => {
  const id=taskId(5);
  await request.put(`/api/contexts/${id}/rating`,{data:{rating:"down"}});
  await page.setViewportSize({width:375,height:812});
  await page.goto(`/contexts/${id}`);
  const group=page.getByRole("group",{name:"Error reasons",exact:true});
  const labels=["Incorrect gene grouping","Incorrect drug grouping","No direct interaction","No indirect interaction",
    "No effects on gene regulation","Not enough evidence","Incorrect directionality","Incorrect supporting evidence"];
  for(const label of labels){
    const button=group.getByRole("button",{name:label,exact:true});
    await expect(button).toBeVisible();
    const box=await button.boundingBox();
    expect(box!.x).toBeGreaterThanOrEqual(0);
    expect(box!.x+box!.width).toBeLessThanOrEqual(375);
    if(await button.getAttribute("aria-pressed")!=="true")await button.click();
    await expect(button).toHaveAttribute("aria-pressed","true");
  }
  await page.reload();
  for(const label of labels)await expect(group.getByRole("button",{name:label,exact:true})).toHaveAttribute("aria-pressed","true");
  expect((await (await request.get(`/api/contexts/${id}`)).json()).error_reasons).toHaveLength(8);
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBeTruthy();
  await page.screenshot({path:"outputs/error-reasons-mobile.png",fullPage:true});
});
