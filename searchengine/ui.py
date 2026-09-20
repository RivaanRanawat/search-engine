"""Search UI served by the engine itself.

Deliberately one self-contained HTML string with no build step, no CDN and
no framework: the engine should be runnable with `python3 -m
searchengine.server` and immediately usable in a browser. Everything it
demonstrates is a real engine feature — autocomplete from the query log,
"did you mean" correction, quoted phrase search, and the ranking mode
actually used for the response.
"""

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Search</title>
<style>
  :root{--fg:#1f2328;--muted:#59636e;--line:#d8dee4;--accent:#0969da;--bg:#fff}
  @media (prefers-color-scheme:dark){
    :root{--fg:#e6edf3;--muted:#9198a1;--line:#30363d;--accent:#4493f8;--bg:#0d1117}}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
    font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
  header{border-bottom:1px solid var(--line);padding:16px 20px;position:sticky;top:0;background:var(--bg)}
  .wrap{max-width:820px;margin:0 auto}
  .row{display:flex;gap:8px;position:relative}
  input[type=search]{flex:1;padding:10px 14px;font-size:16px;border:1px solid var(--line);
    border-radius:8px;background:var(--bg);color:var(--fg)}
  button{padding:10px 16px;border:1px solid var(--line);border-radius:8px;background:var(--accent);
    color:#fff;font-size:15px;cursor:pointer}
  select{padding:10px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--fg)}
  #sugg{position:absolute;top:46px;left:0;right:0;background:var(--bg);border:1px solid var(--line);
    border-radius:8px;z-index:5;overflow:hidden;display:none}
  #sugg div{padding:8px 14px;cursor:pointer}
  #sugg div:hover,#sugg div.sel{background:rgba(127,127,127,.15)}
  main{padding:20px}
  .meta{color:var(--muted);font-size:13px;margin:10px 0 18px}
  .hit{padding:14px 0;border-bottom:1px solid var(--line)}
  .hit .id{color:var(--muted);font-size:12px}
  .hit .snip{margin-top:4px}
  .hit a{color:var(--accent);text-decoration:none;font-weight:600}
  mark{background:rgba(255,212,0,.35);color:inherit;padding:0 1px;border-radius:2px}
  .dym{margin:6px 0 0;font-size:15px}
  .dym a{color:var(--accent);cursor:pointer;text-decoration:underline}
  .tag{display:inline-block;border:1px solid var(--line);border-radius:999px;
    padding:1px 8px;font-size:12px;color:var(--muted);margin-left:6px}
  .empty{color:var(--muted);padding:30px 0}
</style></head><body>
<header><div class="wrap">
  <div class="row">
    <input id="q" type="search" placeholder='Search 8.8M passages — try "cost of living" in quotes'
           autocomplete="off" autofocus>
    <select id="mode" title="ranking mode">
      <option value="1">auto</option><option value="0">bm25 only</option><option value="ce">precision (slow)</option>
    </select>
    <button id="go">Search</button>
    <div id="sugg"></div>
  </div>
  <p class="dym" id="dym"></p>
</div></header>
<main><div class="wrap"><div id="meta" class="meta"></div><div id="out"></div></div></main>
<script>
const $=s=>document.querySelector(s), q=$("#q"), out=$("#out"), sugg=$("#sugg");
let selIdx=-1, timer=null;
const esc=s=>s.replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function highlight(text, terms){
  let h=esc(text);
  for(const t of terms){ if(t.length<2) continue;
    h=h.replace(new RegExp("\\\\b("+t.replace(/[.*+?^${}()|[\\]\\\\]/g,"\\\\$&")+")","gi"),"<mark>$1</mark>"); }
  return h;
}
async function search(){
  const text=q.value.trim(); if(!text) return;
  sugg.style.display="none";
  const t0=performance.now();
  const r=await fetch(`/search?q=${encodeURIComponent(text)}&k=10&rerank=${$("#mode").value}`);
  const d=await r.json(); const rt=Math.round(performance.now()-t0);
  const terms=text.replace(/"/g,"").split(/\\s+/).filter(Boolean);
  $("#meta").innerHTML=`${d.hits.length} results · server ${d.took_ms} ms · round-trip ${rt} ms`
    +`<span class="tag">${d.ranking}</span>`+(d.phrase?`<span class="tag">phrase: ${esc(d.phrase.phrases.join(", "))}</span>`:"");
  $("#dym").innerHTML=d.did_you_mean
    ? `Did you mean <a onclick="q.value=${JSON.stringify(d.did_you_mean)};search()">${esc(d.did_you_mean)}</a>?` : "";
  out.innerHTML = d.hits.length ? d.hits.map(h=>`<div class="hit">
      <div class="id">${h.url?`<a href="${esc(h.url)}" target="_blank" rel="noopener">${esc(h.title||h.url)}</a><br>`:""}doc ${h.pid} · score ${h.score}</div>
      <div class="snip">${highlight(h.snippet||"",terms)}</div></div>`).join("")
    : `<p class="empty">No results.</p>`;
}
async function suggest(){
  const text=q.value; if(!text.trim()){sugg.style.display="none";return;}
  const d=await (await fetch(`/suggest?q=${encodeURIComponent(text)}&k=6`)).json();
  if(!d.suggestions.length){sugg.style.display="none";return;}
  selIdx=-1;
  sugg.innerHTML=d.suggestions.map(s=>`<div>${esc(s)}</div>`).join("");
  sugg.style.display="block";
  [...sugg.children].forEach(el=>el.onclick=()=>{q.value=el.textContent;search()});
}
q.addEventListener("input",()=>{clearTimeout(timer);timer=setTimeout(suggest,90)});
q.addEventListener("keydown",e=>{
  const items=[...sugg.children];
  if(e.key==="ArrowDown"||e.key==="ArrowUp"){
    if(!items.length)return; e.preventDefault();
    selIdx=(selIdx+(e.key==="ArrowDown"?1:-1)+items.length)%items.length;
    items.forEach((el,i)=>el.classList.toggle("sel",i===selIdx));
    q.value=items[selIdx].textContent;
  } else if(e.key==="Enter"){ sugg.style.display="none"; search(); }
  else if(e.key==="Escape"){ sugg.style.display="none"; }
});
$("#go").onclick=search;
document.addEventListener("click",e=>{if(!sugg.contains(e.target)&&e.target!==q)sugg.style.display="none"});
</script></body></html>
"""
