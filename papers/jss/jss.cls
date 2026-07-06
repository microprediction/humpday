%% Journal of Statistical Software (JSS) LaTeX Class
%% Minimal version for HumpDay paper

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{jss}[2023/01/01 Journal of Statistical Software]

% Load base class
\LoadClass[11pt,a4paper]{article}

% Required packages
\RequirePackage{amsmath,amsfonts,amssymb}
\RequirePackage{graphicx}
\RequirePackage{color}
\RequirePackage{url}
\RequirePackage{hyperref}
\RequirePackage{booktabs}
\RequirePackage{fancyhdr}
\RequirePackage{geometry}

% Page geometry
\geometry{
  left=3cm,
  right=3cm,
  top=3cm,
  bottom=3cm
}

% Colors
\definecolor{jssblue}{RGB}{32,80,129}

% Hyperlink setup
\hypersetup{
  colorlinks=true,
  linkcolor=jssblue,
  citecolor=jssblue,
  urlcolor=jssblue
}

% Title formatting
\renewcommand{\maketitle}{%
  \begin{center}
    {\LARGE\bfseries \@title \par}
    \vskip 1em
    {\large \@author \par}
    \vskip 0.5em
    {\large \@date \par}
  \end{center}
  \vskip 2em
}

% Section formatting
\renewcommand{\section}{\@startsection{section}{1}{0pt}%
  {-3.5ex plus -1ex minus -.2ex}%
  {2.3ex plus .2ex}%
  {\large\bfseries}}

% Commands for JSS-specific elements
\newcommand{\pkg}[1]{\textbf{#1}}
\newcommand{\code}[1]{\texttt{#1}}
\newcommand{\email}[1]{\texttt{#1}}

% Abstract environment
\renewenvironment{abstract}%
  {\small\quotation\noindent\textbf{Abstract.}~\ignorespaces}%
  {\endquotation}

% Keywords
\newcommand{\Keywords}[1]{\par\vskip 1em\noindent\textbf{Keywords:} #1}
\newcommand{\Plainkeywords}[1]{}

% Address
\newcommand{\Address}[1]{\par\vskip 2em\noindent#1}

% Title commands
\newcommand{\Plaintitle}[1]{}
\newcommand{\Shorttitle}[1]{}

% Listings setup for code
\RequirePackage{listings}
\lstset{
  basicstyle=\ttfamily\small,
  breaklines=true,
  frame=single,
  backgroundcolor=\color{gray!10}
}
