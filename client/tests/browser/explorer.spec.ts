import { test, expect } from "@playwright/test";
const taskId = (i: number) => i.toString(16).padStart(64,"0");

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
