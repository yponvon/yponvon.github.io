// shared by every project page: contents sidebar, current-section highlight, click-to-zoom images
(function () {
  const main = document.querySelector('main');
  const toc = document.querySelector('nav.toc');

  // build the contents list from the h2/h3 headings if the page didn't write one by hand
  if (toc && !toc.querySelector('a')) {
    main.querySelectorAll('h2[id], h3[id]').forEach(h => {
      const a = document.createElement('a');
      a.href = '#' + h.id;
      const num = h.querySelector('.num');
      const title = [...h.childNodes].filter(n => !(n.classList && n.classList.contains('num')))
                                     .map(n => n.textContent).join('').trim();
      if (num && h.tagName === 'H3') a.textContent = num.textContent + ' ' + title;
      else if (num) a.textContent = num.textContent + ' · ' + title;   // e.g. "Part 1 · Taking a photo"
      else a.textContent = title;
      if (h.tagName === 'H3') a.className = 'sub';
      toc.appendChild(a);
    });
  }

  // click any image to view it large; click again (or press Esc) to close
  const box = document.createElement('div');
  box.id = 'lightbox';
  box.innerHTML = '<img alt="">';
  document.body.appendChild(box);
  const big = box.querySelector('img');
  main.querySelectorAll('img').forEach(img => {
    img.loading = 'lazy';
    img.addEventListener('click', () => { big.src = img.src; big.alt = img.alt; box.classList.add('open'); });
  });
  box.addEventListener('click', () => box.classList.remove('open'));
  document.addEventListener('keydown', e => { if (e.key === 'Escape') box.classList.remove('open'); });

  // highlight the current section in the contents list
  if (!toc) return;
  const links = [...toc.querySelectorAll('a')];
  const targets = links.map(a => document.querySelector(a.getAttribute('href')));
  const onScroll = () => {
    let current = 0;
    targets.forEach((t, i) => { if (t && t.getBoundingClientRect().top < 120) current = i; });
    links.forEach((a, i) => a.classList.toggle('active', i === current));
  };
  document.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
})();
